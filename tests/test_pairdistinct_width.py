"""THE CODE WIDTH LAW (2026-10-25): pairdistinct's tier shelf keeps a key's codes in u8 only when its V <= 256, else
u16. With u8 for every key, a code past 255 wrapped onto code - 256 and two groups counted as one (ParamOrderID,
V 397: 9 + 12 = 21 users reported where DuckDB has 9; 23 wrong on the full Q11 board). Here the filtered key holds
400 values (codes 256..399 would wrap onto 0..143) and, in the second query, the other key holds 300; every
returned group's distinct count is DuckDB's, the top values are DuckDB's, in RAM and through the saved shelf."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_FX = None


def _fixture():
    """200K rows: u (20,000 users), b ('' on ~25%, else 400 strings), a2 (2 values), a300 (300 values),
    s50 ('' on ~25%, else 50 strings)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST(hash(i, 9) % 20000 AS BIGINT) AS u,
        CASE WHEN hash(i, 1) % 4 = 0 THEN '' ELSE 'p' || lpad(CAST(hash(i, 2) % 400 AS VARCHAR), 3, '0') END AS b,
        CAST(i % 2 AS INTEGER) AS a2,
        CAST(hash(i, 3) % 300 AS INTEGER) AS a300,
        CASE WHEN hash(i, 4) % 4 = 0 THEN '' ELSE 'q' || lpad(CAST(hash(i, 5) % 50 AS VARCHAR), 2, '0') END AS s50
        FROM range(200000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'pdw_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['u', 'int'], ['b', 'str'], ['a2', 'int'], ['a300', 'int'], ['s50', 'str']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    _FX = (db, con, seg, d)
    return _FX


QS = ["SELECT a2, b, COUNT(DISTINCT u) AS n FROM t WHERE b <> '' GROUP BY a2, b ORDER BY n DESC LIMIT 10",
      "SELECT a300, s50, COUNT(DISTINCT u) AS n FROM t WHERE s50 <> '' GROUP BY a300, s50 ORDER BY n DESC LIMIT 10"]


def _check(con, q, rows):
    """every group real with its exact distinct count; the top values DuckDB's"""
    k1, k2 = re.match(r'SELECT (\w+), (\w+),', q).groups()
    f = re.search(r'WHERE (.*?) GROUP BY', q).group(1)
    want = con.execute(q).fetchall()
    assert sorted([r[-1] for r in rows], reverse=True) == sorted([w[-1] for w in want], reverse=True), \
        (q, [r[-1] for r in rows], [w[-1] for w in want])
    for r in rows:
        n = con.execute('SELECT COUNT(DISTINCT u) FROM t WHERE %s AND %s = ? AND %s = ?' % (f, k1, k2),
                        [r[0], r[1]]).fetchone()[0]
        assert n == r[-1], (q, r, n)


def test_wide_codes_count_their_own_groups():
    db, con, seg, d = _fixture()
    import sqlglot, wdb_pairdistinct as PD, wdb_sidecar
    assert int(seg.cols['b']['V']) > 256 and int(seg.cols['a300']['V']) > 256, \
        (seg.cols['b']['V'], seg.cols['a300']['V'])
    tiered = [c for c in ('b', 's50') if seg.cols[c].get('code_enc') not in (8, 9)]
    assert tiered, 'both filtered keys are sparse: the tier shelf is never built'
    births0 = wdb_sidecar.births_on
    ran = 0
    try:
        for births in (False, True):            # in RAM, then saved to the file and read back from it
            wdb_sidecar.births_on = lambda _d, _b=births: _b
            for q in QS:
                spec = PD.detect(seg, sqlglot.parse_one(q), {})
                assert spec is not None, q
                rows, _names = PD.execute(seg, spec)
                _check(con, q, [tuple(r) for r in rows])
                ran += 1
    finally:
        wdb_sidecar.births_on = births0
    assert ran == 4, ran
    shelves = [f for f in os.listdir(d) if f.startswith('tier2__')]
    assert any(f.endswith('__w12.bin') for f in shelves), shelves     # a2 u8, b u16: the widths in the name
