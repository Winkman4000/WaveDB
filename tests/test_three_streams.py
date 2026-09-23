"""THE THREE STREAMS (Jackson, 2026-09-23): a chunked front-coded dictionary stored as headers
(<cp sl> per entry), a start-of-character mask (one bit per text byte) and the text. Length in
characters = the set bits of an entry's mask, carried through the shared prefix; no text byte is
read. The same data written the old way (WDB_FC3 off) must read back identically everywhere."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql, wdb_strings, wdb_kernels as WK
from wdb_engine import Segment

TMP = tempfile.gettempdir()
_FIX = None


def _fixture():
    """one table encoded twice: three streams, and the interleaved frames"""
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(11)
    vals = set()
    while len(vals) < 70000:                     # past FC_THRESHOLD (50000): front-coded, chunked
        k = int(rng.integers(0, 3000)); x = int(rng.integers(0, 10 ** 6)); r = int(rng.integers(0, 9))
        if r == 4 and rng.random() > 0.03:
            r = 0                                # a few long strings, not megabytes of them
        if r == 0: v = 'http://www.site%d.com/p/%d' % (k, x)
        elif r == 1: v = 'https://пример%d.рф/страница/%d' % (k, x)
        elif r == 2: v = 'Google Поиск %d 日本語 %d' % (k, x)
        elif r == 3: v = 'https://www.google.com/search?q=%d' % x
        elif r == 4: v = 'x' * int(rng.integers(4000, 9000)) + str(x)      # past the old 4 KB carry
        elif r == 5: v = 'emoji 😀%d' % x                                      # four-byte characters
        elif r == 6: v = 'http://site%d.com/a\n%d' % (k, x)
        elif r == 7: v = 'Ё%d' % x
        else: v = ''
        vals.add(v)
    vals = sorted(vals)
    n = 150000
    ref = np.array(vals, dtype=object)[rng.integers(0, len(vals), n)]
    df = pd.DataFrame({'ref': ref, 'g': rng.integers(0, 40, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/ts_{t}.parquet'; w3 = f'{TMP}/ts3_{t}.wdb'; w1 = f'{TMP}/ts1_{t}.wdb'
    df.to_parquet(pq, index=False)
    wdb_encode.encode(pq, w3)
    prev = wdb_encode.FC3_DICT; wdb_encode.FC3_DICT = False
    try:
        wdb_encode.encode(pq, w1)
    finally:
        wdb_encode.FC3_DICT = prev
    _FIX = (w3, w1, pq, df)
    return _FIX


def _dict(seg, col):
    V0 = int(seg.cols[col].get('n_dict') or seg.cols[col]['V'])
    return [seg.fetch(col, i) for i in range(V0)]


def test_layouts_and_rejoin_is_exact():
    w3, w1, pq, df = _fixture()
    s3, s1 = Segment(w3), Segment(w1)
    c3, c1 = s3.cols['ref'], s1.cols['ref']
    assert c3.get('fc3') and not c1.get('fc3') and c3['nch'] == c1['nch'] >= 3
    assert 'chunk_foff' not in c3                     # no reader may misread the new layout
    for j in range(c3['nch']):
        assert s3.fc_chunk(c3, j, as_bytes=True) == s1.fc_chunk(c1, j, as_bytes=True), j


def test_mask_is_the_start_of_every_character():
    w3, w1, pq, df = _fixture()
    s3 = Segment(w3); c = s3.cols['ref']
    for j in range(c['nch']):
        t = s3.fc_part(c, j, 't'); m = s3.fc_part(c, j, 'm')
        bits = np.unpackbits(m, bitorder='little')[:t.size].astype(bool)
        assert np.array_equal(bits, (t & 0xC0) != 0x80), j
        assert not np.unpackbits(m, bitorder='little')[t.size:].any(), j   # padding is zero


def test_lengths_equal_definition_and_old_layout():
    w3, w1, pq, df = _fixture()
    s3, s1 = Segment(w3), Segment(w1)
    d = _dict(s1, 'ref')
    exp_c = np.array([len(v.decode('utf-8')) for v in d]); exp_b = np.array([len(v) for v in d])
    assert max(exp_b) > 4096                           # the long strings are really there
    assert np.array_equal(np.asarray(s3.dict_charlens('ref'))[:len(d)], exp_c)
    assert np.array_equal(np.asarray(s1.dict_charlens('ref'))[:len(d)], exp_c)
    assert np.array_equal(np.asarray(s3.dict_bytelens('ref'))[:len(d)], exp_b)


def test_fetch_values_and_contains_equal_old_layout():
    w3, w1, pq, df = _fixture()
    s3, s1 = Segment(w3), Segment(w1)
    assert _dict(s3, 'ref') == _dict(s1, 'ref')
    V = int(s3.cols['ref']['V']); rng = np.random.default_rng(5)
    codes = rng.integers(0, V, 3000)
    assert s3.values_at('ref', codes) == s1.values_at('ref', codes)
    for n1, n2 in ((b'google', b''), ('Поиск'.encode(), b''), ('😀'.encode(), b''), (b'xxxx', b''),
                   (b'\n', b''), (b'http', b'/p/'), (b'zzzz', b'')):
        a = wdb_strings.identify_contains(s3, 'ref', n1, n2); b = wdb_strings.identify_contains(s1, 'ref', n1, n2)
        assert np.array_equal(a, b), (n1, n2)


def test_queries_match_duck():
    w3, w1, pq, df = _fixture()
    con = duckdb.connect()
    for sql in ["SELECT COUNT(*) FROM TBL WHERE ref LIKE '%google%'",
                "SELECT g, AVG(length(ref)), COUNT(*) FROM TBL WHERE ref <> '' GROUP BY g ORDER BY g",
                "SELECT COUNT(*) FROM TBL WHERE ref NOT LIKE '%Поиск%'",
                "SELECT MIN(ref), MAX(length(ref)) FROM TBL",
                "SELECT REGEXP_REPLACE(ref, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, AVG(length(ref)) AS l, "
                "COUNT(*) AS c, MIN(ref) FROM TBL WHERE ref <> '' GROUP BY k HAVING COUNT(*) > 30 ORDER BY l DESC, k LIMIT 25"]:
        duck = con.execute(sql.replace('TBL', f"'{pq}'")).fetchall()
        rows, _ = wdb_sql.execute(Segment(w3), sql.replace('TBL', 'tbl'))
        norm = lambda rs: sorted(tuple(round(float(x), 6) if isinstance(x, float) else
                                       (x.decode() if isinstance(x, bytes) else x) for x in r) for r in rs)
        assert norm(rows) == norm(duck), sql
