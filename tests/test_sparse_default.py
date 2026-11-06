"""THE SPARSE DEFAULT (c129, 2026-11-06, the tuning sweep): reads that answer `col <> ''` from a sparse column's planes
(the rows whose value differs from the stored default) are right only when that default IS ''. On a column whose most
common value is something else they dropped its rows (ClickBench's BrowserLanguage: 'S0', no '' at all). Every shape
those reads serve, on a sparse column with default 'x' that also holds '' -- plus a text column with no '' (its code 0
is a real value), a column with NULLs, the trunc lane's dropped filter and its ORDER BY, pairdistinct's ORDER BY.
Every answer DuckDB's (ties at a LIMIT cut: the cut's values exact, every group's own value true)."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_FX = None
N = 400000
NREP = 12                       # repeated (w, ip) pairs at the end of the table, counts 2..13


def _fixture():
    """N rows in time order: tm one tick per 4 rows (a staircase), d its date; per block of 4 rows: n (no '': 'n0'..'n39'),
    nn (NULL 20%, '' 10%, 'm0'..'m6'); per row: s (sparse: 'x' 80%, '' 5%, 'v0'..'v999' at random -- scattered, so the
    planes are the smallest storage; in blocks the runs compress better and s is not stored sparse), k (3 values),
    c (300: the selector), f (flag), u (users), w/ip near-unique pairs with NREP repeated pairs of distinct counts at
    the end, g (NULL 40% -- the biggest group, so a read that lets NULL through `<>` shows it at the top -- 'g0' 30%,
    'g1' 20%, 'g2' 10%); and a 3-row table dm (k, nm) to join on k"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from wdb_db import Database
    i = np.arange(N, dtype=np.int64)
    b = i // 4
    rng = np.random.default_rng(7)
    h = rng.random(N); hv = rng.integers(0, 1000, N)
    s = np.where(h < 0.80, 'x', np.where(h < 0.85, '', np.char.add('v', hv.astype(str))))
    hg = rng.random(N)
    g = [None if x < 0.4 else ('g0' if x < 0.7 else ('g1' if x < 0.9 else 'g2')) for x in hg.tolist()]
    n = np.char.add('n', ((b * 48271) % 40).astype(str))
    h3 = (b * 69621) % 10
    nn = [None if x < 2 else ('' if x == 2 else 'm%d' % (y % 7)) for x, y in zip(h3.tolist(), (b * 48271).tolist())]
    k = ((i * 7919) % 3).astype(np.int32)
    c = ((i * 2654435761) % 300).astype(np.int32)
    f = np.where((i * 7919) % 10 < 8, 0, 1).astype(np.int32)
    u = ((i * 48271) % 20011).astype(np.int64)
    w = i.copy(); ip = ((i * 7) % 1000003).astype(np.int64)
    # the repeated pairs: pair j (0..NREP-1) holds j+2 rows; even j in the default 'x', odd in 'v1', and the biggest
    # pair in '' (so `<> ''` must drop it, and the default rows must stay)
    r0 = N - sum(j + 2 for j in range(NREP)); pos = r0
    for j in range(NREP):
        for _ in range(j + 2):
            w[pos] = -(j + 1); ip[pos] = 77
            s[pos] = '' if j == NREP - 1 else ('x' if j % 2 == 0 else 'v1')
            pos += 1
    tm = (np.datetime64('2013-07-01T00:00:00') + (b * 7).astype('timedelta64[s]')).astype('datetime64[s]')
    d = tm.astype('datetime64[D]')
    tbl = pa.table({'id': i, 'tm': pa.array(tm, pa.timestamp('s')), 'd': pa.array(d, pa.date32()), 'k': k, 'c': c,
                    'f': f, 'u': u, 'w': w, 'ip': ip, 's': pa.array(s.tolist(), pa.string()),
                    'n': pa.array(n.tolist(), pa.string()), 'nn': pa.array(nn, pa.string()),
                    'g': pa.array(g, pa.string())})
    dd = os.path.join(tempfile.gettempdir(), 'spd_%s' % uuid.uuid4().hex[:8]); os.makedirs(dd, exist_ok=True)
    try:
        import helpers as _H; _H.register_dir(dd)
    except Exception:
        pass
    p = os.path.join(dd, 't.parquet'); pq.write_table(tbl, p)
    pdm = os.path.join(dd, 'dm.parquet')
    pq.write_table(pa.table({'k': np.arange(3, dtype=np.int32), 'nm': pa.array(['k0', 'k1', 'k2'], pa.string())}), pdm)
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % p)
    con.execute("CREATE VIEW dm AS SELECT * FROM read_parquet('%s')" % pdm)
    db = Database.create(dd)
    db.cat.add_table('t', [['id', 'int'], ['tm', 'datetime'], ['d', 'date'], ['k', 'int'], ['c', 'int'], ['f', 'int'],
                           ['u', 'int'], ['w', 'int'], ['ip', 'int'], ['s', 'str'], ['n', 'str'], ['nn', 'str'],
                           ['g', 'str']])
    saved = {k9: os.environ.pop(k9, None) for k9 in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:                                         # the runner's sequence levers would store tm raw, not a staircase
        wdb_encode.encode(p, os.path.join(dd, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    db.cat.add_table('dm', [['k', 'int'], ['nm', 'str']])
    wdb_encode.encode(pdm, os.path.join(dd, 'dm_0.wdb'))
    db.cat.add_segment('dm', 'dm_0.wdb')
    _FX = (db, con)
    return _FX


def _rows(r):
    out = []
    for x in (r[0] if isinstance(r, tuple) else r):
        t = []
        for v in x:
            if hasattr(v, 'isoformat'):
                v = str(v)[:19]
            elif isinstance(v, float):
                v = round(v, 6)
            elif isinstance(v, (bytes, bytearray)):
                v = v.decode()
            t.append(v)
        out.append(tuple(t))
    return out


def _items(sel):
    """the projection list of a SELECT, split at its top-level commas"""
    out, depth, cur = [], 0, ''
    for ch in sel:
        depth += (ch == '(') - (ch == ')')
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ''
        else:
            cur += ch
    return out + [cur.strip()]


def _legal(q, got, con):
    """exact, or -- a DESC LIMIT on an aggregate -- the cut's ordering values exact and every row's own values true"""
    want = _rows(con.execute(q).fetchall())
    if got == want:
        return True
    m = re.search(r' ORDER BY (\w+) DESC LIMIT', q)
    if m is None or len(got) != len(want):
        return False
    items = _items(q[len('SELECT '):q.index(' FROM t ')])
    oi = next(i for i, it in enumerate(items) if it.endswith(' AS ' + m.group(1)))
    if [x[oi] for x in got] != [x[oi] for x in want]:
        return False
    full = set(_rows(con.execute(re.sub(r' ORDER BY .*$', '', q)).fetchall()))
    return all(x in full for x in got)


S = "s <> ''"
SHAPES = [
    # the filter column's default is 'x': its rows must STAY
    "SELECT k, s, COUNT(*) AS cc FROM t WHERE %s GROUP BY k, s ORDER BY cc DESC LIMIT 5" % S,           # Q14
    "SELECT s, COUNT(*) AS cc FROM t WHERE %s GROUP BY s ORDER BY cc DESC LIMIT 5" % S,                 # Q12
    "SELECT s, COUNT(DISTINCT u) AS du FROM t WHERE %s GROUP BY s ORDER BY du DESC LIMIT 5" % S,        # Q13
    "SELECT k, s, COUNT(DISTINCT u) AS du FROM t WHERE %s GROUP BY k, s ORDER BY du DESC LIMIT 5" % S,  # Q11
    "SELECT w, ip, COUNT(*) AS cc, SUM(f), AVG(k) FROM t WHERE %s GROUP BY w, ip ORDER BY cc DESC LIMIT 10" % S,  # Q31
    "SELECT s FROM t WHERE %s ORDER BY tm LIMIT 10" % S,                                                 # Q24
    "SELECT s FROM t WHERE %s ORDER BY s LIMIT 10" % S,                                                  # Q25
    "SELECT s FROM t WHERE %s ORDER BY tm, s LIMIT 10" % S,                                              # Q26
    "SELECT u, extract(minute FROM tm) AS m, s, COUNT(*) AS cc FROM t GROUP BY u, m, s ORDER BY cc DESC LIMIT 10",  # Q18
    # a text column with no '' (code 0 is 'n0', a real value): `<> ''` keeps every row
    "SELECT n FROM t WHERE n <> '' ORDER BY tm LIMIT 10",
    "SELECT n, COUNT(*) AS cc FROM t WHERE n <> '' GROUP BY n ORDER BY cc DESC LIMIT 5",
    "SELECT k, n, COUNT(*) AS cc FROM t WHERE n <> '' GROUP BY k, n ORDER BY cc DESC LIMIT 5",
    # NULLs: NULL <> '' is not true
    "SELECT nn FROM t WHERE nn <> '' ORDER BY tm LIMIT 10",
    "SELECT nn, COUNT(*) AS cc FROM t WHERE nn <> '' GROUP BY nn ORDER BY cc DESC LIMIT 5",
    "SELECT k, nn, COUNT(*) AS cc FROM t WHERE nn <> '' GROUP BY k, nn ORDER BY cc DESC LIMIT 5",
    "SELECT k, nn, COUNT(*) AS cc FROM t WHERE nn <> 'zz' GROUP BY k, nn ORDER BY cc DESC LIMIT 5",
    "SELECT k, nn, COUNT(*) AS cc FROM t GROUP BY k, nn ORDER BY cc DESC LIMIT 5",        # a NULL group is a group
    "SELECT nn, COUNT(*) AS cc FROM t WHERE nn <> 'zz' GROUP BY nn ORDER BY cc DESC LIMIT 5",
    "SELECT w, ip, COUNT(*) AS cc, SUM(f), AVG(k) FROM t WHERE nn <> '' GROUP BY w, ip ORDER BY cc DESC LIMIT 10",
    "SELECT nn FROM t WHERE nn <> 'zz' AND f = 1 ORDER BY tm LIMIT 10",
    "SELECT nn FROM t WHERE f = 1 AND nn <> '' ORDER BY tm LIMIT 10",
    # NULL is g's biggest group: `<>` must drop it, GROUP BY must keep it
    "SELECT k, g, COUNT(*) AS cc FROM t WHERE g <> 'g2' GROUP BY k, g ORDER BY cc DESC LIMIT 3",
    "SELECT k, g, COUNT(*) AS cc FROM t WHERE g <> 'zz' GROUP BY k, g ORDER BY cc DESC LIMIT 3",
    "SELECT k, g, COUNT(*) AS cc FROM t GROUP BY k, g ORDER BY cc DESC LIMIT 3",
    "SELECT g, COUNT(*) AS cc FROM t WHERE g <> 'g2' GROUP BY g ORDER BY cc DESC LIMIT 3",
    "SELECT g, COUNT(DISTINCT u) AS du FROM t WHERE g <> 'g2' GROUP BY g ORDER BY du DESC LIMIT 3",
    "SELECT w, ip, COUNT(*) AS cc, SUM(f), AVG(k) FROM t WHERE g <> 'g2' GROUP BY w, ip ORDER BY cc DESC LIMIT 10",
    "SELECT w, ip, COUNT(*) AS cc, SUM(f), AVG(k) FROM t WHERE g <> 'zz' GROUP BY w, ip ORDER BY cc DESC LIMIT 10",
    "SELECT g FROM t WHERE g <> 'g0' ORDER BY tm LIMIT 10",
    "SELECT g, tm FROM t WHERE g <> 'g0' ORDER BY g DESC, tm LIMIT 10",
    "SELECT g, COUNT(*) AS cc FROM t GROUP BY g ORDER BY cc DESC LIMIT 3",
    "SELECT k, COUNT(*) AS cc FROM t WHERE id >= 1000 AND g <> 'g2' GROUP BY k ORDER BY cc DESC LIMIT 3",
    "SELECT k, COUNT(*) AS cc FROM t WHERE id >= 1000 AND g < 'g2' GROUP BY k ORDER BY cc DESC LIMIT 3",
    "SELECT k, COUNT(*) AS cc FROM t WHERE id >= 1000 AND nn <> '' GROUP BY k ORDER BY cc DESC LIMIT 3",
    "SELECT REGEXP_REPLACE(g, '^g', 'h') AS r, COUNT(*) AS cc, MIN(g) FROM t GROUP BY r ORDER BY cc DESC LIMIT 5",
    "SELECT REGEXP_REPLACE(g, '^g', 'h') AS r, AVG(length(g)) AS l, COUNT(*) AS cc, MIN(g) FROM t WHERE g <> '' "
    "GROUP BY r ORDER BY cc DESC LIMIT 5",
    # a literal the column does not hold: `<>` keeps every row, the default's too
    "SELECT w, ip, COUNT(*) AS cc, SUM(f), AVG(k) FROM t WHERE s <> 'zz' GROUP BY w, ip ORDER BY cc DESC LIMIT 10",
    "SELECT k, s, COUNT(*) AS cc FROM t WHERE s <> 'zz' GROUP BY k, s ORDER BY cc DESC LIMIT 5",
]
JOINS = [
    # an unordered LIMIT over a join: every row returned must be a row of the answer (NULL <> '' is not true)
    "SELECT t.id, dm.nm, t.nn FROM t JOIN dm ON t.k = dm.k WHERE t.nn <> '' LIMIT 20",
    "SELECT t.id, dm.nm, t.nn FROM t JOIN dm ON t.k = dm.k WHERE t.nn <> 'zz' LIMIT 20",
    "SELECT t.id, dm.nm, t.s FROM t JOIN dm ON t.k = dm.k WHERE t.s <> 'x' LIMIT 20",
    "SELECT t.id, dm.nm, t.s FROM t JOIN dm ON t.k = dm.k WHERE t.s <> '' LIMIT 20",
    "SELECT t.id, dm.nm, t.g FROM t JOIN dm ON t.k = dm.k WHERE t.g <> 'g2' LIMIT 20",
]
W = "c = 62 AND d >= '2013-07-02' AND d <= '2013-07-05' AND f = 0"
TRUNC = [
    # the dashboard's minute buckets with a `<> ''` filter (it was dropped), and ORDER BY something else than the bucket
    "SELECT DATE_TRUNC('minute', tm) AS M, COUNT(*) AS cc FROM t WHERE %s AND s <> '' GROUP BY DATE_TRUNC('minute', tm) "
    "ORDER BY DATE_TRUNC('minute', tm) LIMIT 10 OFFSET 3" % W,
    "SELECT DATE_TRUNC('minute', tm) AS M, COUNT(*) AS cc FROM t WHERE %s AND n <> '' GROUP BY DATE_TRUNC('minute', tm) "
    "ORDER BY M LIMIT 10 OFFSET 3" % W,
    "SELECT DATE_TRUNC('hour', tm) AS M, COUNT(*) AS cc FROM t WHERE %s GROUP BY DATE_TRUNC('hour', tm) "
    "ORDER BY cc DESC LIMIT 5" % W,
    "SELECT DATE_TRUNC('minute', tm) AS M, COUNT(*) AS cc FROM t WHERE %s GROUP BY DATE_TRUNC('minute', tm) "
    "ORDER BY 1 LIMIT 10 OFFSET 3" % W,
]


def test_the_columns_are_what_the_test_needs():
    """s is stored sparse with default 'x' (else the shapes below test nothing); n holds no ''; nn holds NULLs"""
    import wdb_join as WJ, wdb_wherescan as WS, wdb_policies as P
    db, _ = _fixture()
    seg, _x = WJ._solo_segment(db, 't')
    cs = seg.cols['s']
    assert cs.get('code_enc') in (8, 9), ('s is not sparse', cs.get('code_enc'))
    assert not P.sparse_default_is(seg, 's', ''), 'the default of s should be x'
    assert P.sparse_default_is(seg, 's', 'x')
    assert WS._code_of(seg, 'n', '') is None
    assert seg.cols['nn'].get('has_null')


def test_every_shape_is_duckdbs():
    import controller
    db, con = _fixture()
    bad = []
    for q in SHAPES + TRUNC:
        got = _rows(db.run(q))
        if not _legal(q, got, con):
            bad.append((controller._SERVED[0], q, got[:3], _rows(con.execute(q).fetchall())[:3]))
    assert not bad, bad


def test_every_join_row_is_an_answer_row():
    import controller
    db, con = _fixture()
    bad = []
    for q in JOINS:
        got = _rows(db.run(q))
        full = set(_rows(con.execute(q.replace(' LIMIT 20', '')).fetchall()))
        if len(got) != min(20, len(full)) or not all(x in full for x in got):
            bad.append((controller._SERVED[0], q, [x for x in got if x not in full][:3]))
    assert not bad, bad


def test_pairdistinct_order_by_another_aggregate():
    """ORDER BY COUNT(*) with COUNT(DISTINCT u) projected: the groups are the biggest by rows, not by users. The general
    path cannot yet ORDER BY an aggregate it does not project: it says so (NotImplementedError) -- loud, never wrong"""
    import controller
    db, con = _fixture()
    q = "SELECT k, s, COUNT(DISTINCT u) AS du FROM t WHERE s <> '' GROUP BY k, s ORDER BY COUNT(*) DESC LIMIT 3"
    try:
        got = _rows(db.run(q))
    except NotImplementedError:
        return
    full = _rows(con.execute("SELECT k, s, COUNT(DISTINCT u), COUNT(*) AS cc FROM t WHERE s <> '' GROUP BY k, s "
                             "ORDER BY cc DESC").fetchall())
    top = [x[3] for x in full[:3]]
    byk = {(x[0], x[1]): (x[2], x[3]) for x in full}
    assert len(got) == 3
    assert sorted((byk[(x[0], x[1])][1] for x in got), reverse=True) == top, (got, full[:5])
    assert all(byk[(x[0], x[1])][0] == x[2] for x in got)


if __name__ == '__main__':
    for nm, fn in list(globals().items()):
        if nm.startswith('test_'):
            fn(); print('PASS', nm)
