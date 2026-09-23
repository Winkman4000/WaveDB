"""THE FILTER BEFORE THE READ (Jackson, 2026-09-23): a LIKE that runs after other filters decides
only the distinct codes still alive, not its whole dictionary. identify_contains_at must equal the
plain definition on any code set, on both dictionary layouts; queries must equal DuckDB whichever
road the LIKE takes; NULL NOT LIKE x is not true."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql, wdb_strings, wdb_wherescan, controller
from wdb_engine import Segment
from wdb_db import Database

TMP = tempfile.gettempdir()
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(21)
    vals = set()
    while len(vals) < 70000:
        k = int(rng.integers(0, 3000)); x = int(rng.integers(0, 10 ** 6)); r = int(rng.integers(0, 6))
        if r == 0: v = 'http://www.site%d.com/p/%d' % (k, x)
        elif r == 1: v = 'https://www.google.com/search?q=%d' % x
        elif r == 2: v = 'http://mail.google.%d.ru/%d' % (k, x)
        elif r == 3: v = 'Google Поиск %d' % x
        elif r == 4: v = 'https://пример%d.рф/%d' % (k, x)
        else: v = ''
        vals.add(v)
    vals = sorted(vals); n = 200000
    ref = np.array(vals, dtype=object)[rng.integers(0, len(vals), n)]
    ref[rng.random(n) < 0.01] = None                               # a NULL bin
    df = pd.DataFrame({'ref': ref, 'g': rng.integers(0, 40, n).astype(np.int64),
                       'h': rng.integers(0, 7, n).astype(np.int64),
                       'ts': pd.to_datetime(1_600_000_000 + np.arange(n), unit='s')})   # a staircase: time order IS file order
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/ff_{t}.parquet'
    df.to_parquet(pq, index=False)
    dbs = []
    for fc3 in (True, False):                                      # one database per layout
        d = f'{TMP}/ffdb{int(fc3)}_{t}'
        db = Database.create(d)
        db.cat.add_table('tbl', [['ref', 'string'], ['g', 'int'], ['h', 'int'], ['ts', 'timestamp']])
        prev = wdb_encode.FC3_DICT; wdb_encode.FC3_DICT = fc3
        try:
            wdb_encode.encode(pq, os.path.join(d, 'tbl_0.wdb'))
        finally:
            wdb_encode.FC3_DICT = prev
        db.cat.add_segment('tbl', 'tbl_0.wdb')
        dbs.append((db, os.path.join(d, 'tbl_0.wdb')))
    _FIX = (dbs[0][1], dbs[1][1], pq, dbs[0][0], dbs[1][0])
    return _FIX


def test_identify_at_equals_definition_both_layouts():
    w3, w1, pq, db3, db1 = _fixture()
    for w in (w3, w1):
        seg = Segment(w); c = seg.cols['ref']
        assert c.get('chunked') and c.get('has_null')
        V0 = int(c['n_dict']); V = int(c['V'])
        d = [seg.fetch('ref', i) for i in range(V0)]
        rng = np.random.default_rng(4)
        edges = np.array([0, 1, 127, 128, 129, 16383, 16384, 16385, V0 - 1, V - 1], np.int64)   # V-1 = NULL
        for size in (1, 40, 3000, V0 // 2):
            u = np.unique(np.concatenate([rng.integers(0, V, size), edges[edges < V]]))
            for n1, n2 in ((b'google', b''), ('Поиск'.encode(), b''), (b'http', b'/p/'), (b'zzz', b'')):
                got = wdb_strings.identify_contains_at(seg, 'ref', u, n1, n2)
                exp = [(i < V0) and (n1 in d[i]) and (not n2 or n2 in d[i][d[i].find(n1) + len(n1):]) for i in u.tolist()]
                assert np.array_equal(got, np.array(exp)), (w, size, n1, n2)


def test_touched_chunks():
    w3, w1, pq, db3, db1 = _fixture()
    seg = Segment(w3); c = seg.cols['ref']
    assert wdb_strings.chunks_touched(seg, 'ref', np.array([0, 5, 16383], np.int64)) == (1, c['nch'])
    assert wdb_strings.chunks_touched(seg, 'ref', np.array([0, 16384], np.int64)) == (2, c['nch'])
    # the bill: codes 0 and 5 walk group 0 to entry 5 (6 entries), code 130 walks group 1 to 130 (3)
    bill, touched, nch, walked = wdb_strings.at_cost(seg, 'ref', np.array([0, 5, 130], np.int64))
    V0 = int(c['n_dict'])
    assert (touched, walked) == (1, 9) and abs(bill - (1 / nch + 9 / V0) / 2) < 1e-12
    assert wdb_strings.at_cost(seg, 'ref', np.array([int(c['V']) - 1], np.int64))[3] == 0   # NULL bin: free


def test_queries_match_duck_on_both_roads():
    w3, w1, pq, db3, db1 = _fixture()
    con = duckdb.connect()
    sqls = ["SELECT COUNT(*) FROM TBL WHERE g = 3 AND ref LIKE '%google%'",
            "SELECT h, COUNT(*) FROM TBL WHERE g = 7 AND ref NOT LIKE '%google%' GROUP BY h ORDER BY h",
            "SELECT h, COUNT(*) FROM TBL WHERE ref LIKE '%Поиск%' AND ref NOT LIKE '%.google.%' AND g <> 5 GROUP BY h ORDER BY h",
            "SELECT COUNT(*) FROM TBL WHERE g = 11 AND ref NOT LIKE '%http%'"]
    norm = lambda rs: sorted(tuple(x.decode() if isinstance(x, bytes) else x for x in r) for r in rs)
    prev = wdb_strings.AT_FORCE[0]
    roads = set()
    try:
        for share in ('survivors', 'whole'):     # pin each road in turn
            wdb_strings.AT_FORCE[0] = share
            for sql in sqls:
                for w, db in ((w3, db3), (w1, db1)):
                    wdb_wherescan._LIKE_AT_LAST[0] = None
                    rows = db.run(sql.replace('TBL', 'tbl'))
                    rows = rows[0] if isinstance(rows, tuple) else rows
                    if wdb_wherescan._LIKE_AT_LAST[0] is not None:
                        roads.add(wdb_wherescan._LIKE_AT_LAST[0][-1])
                    assert norm(rows) == norm(con.execute(sql.replace('TBL', f"'{pq}'")).fetchall()), (share, w, sql)
    finally:
        wdb_strings.AT_FORCE[0] = prev
    assert roads == {'survivors', 'whole dictionary'}, roads   # both roads really ran


def test_first_k_in_time_order_decides_only_what_it_meets():
    """Q23's shape: the staircase walk decides the LIKE for the codes it meets, window by window"""
    w3, w1, pq, db3, db1 = _fixture()
    con = duckdb.connect()
    prev = wdb_strings.AT_FORCE[0]
    try:
        for road in ('survivors', 'whole', None):
            wdb_strings.AT_FORCE[0] = road
            for nd in ('google', 'Поиск', 'zzzz'):
                sql = "SELECT * FROM TBL WHERE ref LIKE '%" + nd + "%' ORDER BY ts LIMIT 10"
                for db in (db3, db1):
                    controller._SERVED[0] = None
                    rows = db.run(sql.replace('TBL', 'tbl'))
                    rows = rows[0] if isinstance(rows, tuple) else rows
                    assert controller._SERVED[0] == 'firstk', controller._SERVED[0]
                    duck = con.execute(sql.replace('TBL', f"'{pq}'")).fetchall()
                    got = [tuple(x.decode() if isinstance(x, bytes) else x for x in r[:3]) for r in rows]
                    assert got == [tuple(r[:3]) for r in duck], (road, nd)
    finally:
        wdb_strings.AT_FORCE[0] = prev
