"""THE PAIR USERS read (2026-10-25): SELECT a, b, COUNT(DISTINCT u) ... [WHERE a|b <> lit] GROUP BY a, b ORDER BY
n DESC LIMIT k. THE KEPT CASE: the filter keeps few rows (census), every pair counted exactly over them. THE SELF
CASE: u is a key, every group holds one user. THE NEAR-UNIQUE CASE: a key with N - V tiny; only its repeated
values can make groups of 2+ users (the band count: rows per band of codes minus the band's codes). Every answer
legal against DuckDB (each group's count exact, no group twice, the top values DuckDB's) and served by pairusers."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_FX = None


def _fixture():
    """300K rows: w (near unique: 299,990 values; rows 1000..1009 repeat rows 0..9's w and s -- row 1000 with row
    0's user (a group of one user), rows 1001..1009 with another (nine groups of two)), u (30,000 users), s (12
    values), tx (strings, 40), r (1 on ~0.5% of rows, else 0)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t0 AS SELECT i AS id,
        CAST(hash(i, 9) % 30000 AS BIGINT) AS u,
        CAST(hash(i, 3) % 12 AS INTEGER) AS s,
        'x' || CAST(hash(i, 4) % 40 AS VARCHAR) AS tx,
        CAST(CASE WHEN hash(i, 11) % 200 = 0 THEN 1 ELSE 0 END AS INTEGER) AS r
        FROM range(300000) r(i)""")
    con.execute("""CREATE TABLE t AS SELECT t0.id,
        CAST(CASE WHEN id BETWEEN 1000 AND 1009 THEN hash(id - 1000, 7) ELSE hash(id, 7) END % 9000000000000000000
             AS BIGINT) AS w,
        CASE WHEN id = 1000 THEN (SELECT u FROM t0 x WHERE x.id = 0)
             WHEN id BETWEEN 1001 AND 1009 THEN (SELECT u FROM t0 x WHERE x.id = t0.id - 1000) + 1
             ELSE u END AS u,
        CASE WHEN id BETWEEN 1000 AND 1009 THEN (SELECT s FROM t0 x WHERE x.id = t0.id - 1000) ELSE s END AS s,
        tx, r FROM t0""")
    d = os.path.join(tempfile.gettempdir(), 'pu_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['w', 'int'], ['u', 'int'], ['s', 'int'], ['tx', 'str'], ['r', 'int']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    _FX = (db, con, seg)
    return _FX


SELF_NEAR = [  # the self case
             "SELECT u, s, COUNT(DISTINCT u) AS n FROM t GROUP BY u, s ORDER BY n DESC LIMIT 10",
             "SELECT tx, u, COUNT(DISTINCT u) AS n FROM t WHERE tx <> 'x7' GROUP BY tx, u ORDER BY n DESC LIMIT 10",
             # the near-unique case: the nine two-user groups first, then one-row groups
             "SELECT w, s, COUNT(DISTINCT u) AS n FROM t GROUP BY w, s ORDER BY n DESC LIMIT 10",
             "SELECT s, w, COUNT(DISTINCT u) AS n FROM t WHERE s <> 3 GROUP BY s, w ORDER BY n DESC LIMIT 12",
             "SELECT w, tx, COUNT(DISTINCT u) AS n FROM t GROUP BY w, tx ORDER BY n DESC LIMIT 5"]
KEPT = ["SELECT s, r, COUNT(DISTINCT u) AS n FROM t WHERE r <> 0 GROUP BY s, r ORDER BY n DESC LIMIT 10",
        "SELECT r, tx, COUNT(DISTINCT u) AS n FROM t WHERE r <> 0 GROUP BY r, tx ORDER BY n DESC LIMIT 10",
        "SELECT w, r, COUNT(DISTINCT u) AS n FROM t WHERE r <> 0 GROUP BY w, r ORDER BY n DESC LIMIT 10"]


def _legal(con, q, rows):
    k1, k2 = re.match(r'SELECT (\w+), (\w+),', q).groups()
    w = re.search(r' WHERE (.*?) GROUP BY', q)
    f = (w.group(1) + ' AND ') if w else ''
    want = con.execute(q).fetchall()
    assert len(rows) == len(want), (q, len(rows), len(want))
    assert sorted([r[-1] for r in rows], reverse=True) == sorted([x[-1] for x in want], reverse=True), \
        (q, [r[-1] for r in rows], [x[-1] for x in want])
    assert len({tuple(r[:2]) for r in rows}) == len(rows), q
    for r in rows:
        n = con.execute('SELECT COUNT(DISTINCT u) FROM t WHERE %s%s = ? AND %s = ?' % (f, k1, k2),
                        [r[0], r[1]]).fetchone()[0]
        assert n == r[-1], (q, r, n)


def _run(db, con, qs):
    import controller, wdb_pairusers as PU
    served, hits0 = [], list(PU._HITS)
    for q in qs:
        controller.plan_epoch_bump()
        r = db.run(q)
        _legal(con, q, [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)])
        served.append(controller._SERVED[0])
    controller.plan_epoch_bump()
    return served, [x - y for x, y in zip(PU._HITS, hits0)]


def test_pairusers_self_and_near_unique_legal():
    db, con, seg = _fixture()
    import wdb_pairusers as PU
    assert seg.cols['w'].get('mode') == 2, seg.cols['w']        # the near-unique case reads a mode-2 dictionary
    assert int(seg.N) - int(seg.cols['w']['V']) == 10, (seg.N, seg.cols['w']['V'])
    line0 = PU.KEPT_LINE
    try:
        PU.KEPT_LINE = 5000                     # the fixture's filters keep 25K-290K rows: past this line
        served, hits = _run(db, con, SELF_NEAR)
    finally:
        PU.KEPT_LINE = line0
    assert served == ['pairusers'] * len(SELF_NEAR), served
    assert hits == [2, 3, 0, 0, 0, 0], hits                     # self x2, near-unique x3, nothing else
    top = con.execute(SELF_NEAR[2]).fetchall()
    assert [x[-1] for x in top[:9]] == [2] * 9 and top[9][-1] == 1, top[:10]


def test_pairusers_kept_case_legal():
    db, con, seg = _fixture()
    import wdb_pairusers as PU
    d0 = PU.DENSE_PAIRS
    try:
        for dense in (d0, 0):                    # the stop-early walk's two pair numberings: bincount, one unique
            PU.DENSE_PAIRS = dense
            served, hits = _run(db, con, KEPT)
            assert served == ['pairusers'] * len(KEPT), (dense, served)
            assert hits == [0, 0, 3, 0, 0, 0], (dense, hits)  # w x r: the kept case before the near-unique one
    finally:
        PU.DENSE_PAIRS = d0


def test_pairusers_declines_other_shapes():
    db, con, seg = _fixture()
    import sqlglot, wdb_pairusers as PU
    d0, w0 = PU.DENSE_CASE, PU.WIDE_ON[0]
    PU.DENSE_CASE = 0; PU.WIDE_ON[0] = False                  # past the dense line, the wide case off: no case
    try:
        assert PU.detect(seg, sqlglot.parse_one(
            "SELECT s, tx, COUNT(DISTINCT u) AS n FROM t GROUP BY s, tx ORDER BY n DESC LIMIT 10"), {}) is None
    finally:
        PU.DENSE_CASE = d0; PU.WIDE_ON[0] = w0
    for q in ("SELECT w, s, COUNT(DISTINCT u) AS n FROM t GROUP BY w, s ORDER BY n DESC, s LIMIT 10",   # tie order
              "SELECT w, s, COUNT(DISTINCT u) AS n FROM t WHERE u > 5 GROUP BY w, s ORDER BY n DESC LIMIT 10",
              "SELECT w, s, COUNT(DISTINCT u) AS n FROM t GROUP BY w, s ORDER BY n DESC LIMIT 10 OFFSET 3"):
        assert PU.detect(seg, sqlglot.parse_one(q), {}) is None, q
    # s <> 99 holds no row: it removes nothing -- past the dense line (tx, s) has no case: detect routes, execute
    # declines
    q = "SELECT tx, s, COUNT(DISTINCT u) AS n FROM t WHERE s <> 99 GROUP BY tx, s ORDER BY n DESC LIMIT 10"
    spec = PU.detect(seg, sqlglot.parse_one(q), {})
    spec['case'] = ('ord', False)
    PU.WIDE_ON[0] = False
    try:
        assert spec is not None and PU.execute(seg, spec) is None, spec
    finally:
        PU.WIDE_ON[0] = w0


def test_cheap_kept_count_reads_the_sparse_count():
    """THE CHEAP KEPT COUNT: a sparse column (enc 8 / 9) excluding its default keeps exactly its stored non-default
    rows (on the ClickBench data MobilePhoneModel's e9n, 5,563,212, is the census's count to the row); excluding
    any other value: None (no census); no filter literal held by a row: every row"""
    import wdb_pairusers as PU

    class _Seg:
        N = 1000
        cols = {'x9': {'code_enc': 9, 'e9d': 0, 'e9n': 123, 'V': 5000000},
                'x8': {'code_enc': 8, 'e8d': 4, 'e8n': 77, 'V': 9000000}}
    assert PU._kept_count(_Seg, 'x9', 0) == 123
    assert PU._kept_count(_Seg, 'x9', 3) is None
    assert PU._kept_count(_Seg, 'x8', 4) == 77
    assert PU._kept_count(_Seg, 'x8', 0) is None
    assert PU._kept_count(_Seg, 'x9', None) == 1000


DENSE = ["SELECT s, tx, COUNT(DISTINCT u) AS n FROM t GROUP BY s, tx ORDER BY n DESC LIMIT 10",
         "SELECT s, tx, COUNT(DISTINCT u) AS n FROM t WHERE tx <> 'x7' GROUP BY s, tx ORDER BY n DESC LIMIT 10",
         "SELECT tx, s, COUNT(DISTINCT u) AS n FROM t WHERE tx <> 'x3' GROUP BY tx, s ORDER BY n DESC LIMIT 10",
         "SELECT tx, s, COUNT(DISTINCT u) AS n FROM t WHERE s <> 4 GROUP BY tx, s ORDER BY n DESC LIMIT 100",
         "SELECT s, r, COUNT(DISTINCT u) AS n FROM t WHERE s <> 99 GROUP BY s, r ORDER BY n DESC LIMIT 7"]


def test_pairusers_dense_case_legal():
    """THE DENSE CASE: no filter, a filter on either key, a literal no row holds, and LIMIT 100 (two 64-pair
    rounds at least) -- every answer legal against DuckDB; the kept line lowered so the kept case stands aside"""
    db, con, seg = _fixture()
    import wdb_pairusers as PU
    line0 = PU.KEPT_LINE
    try:
        PU.KEPT_LINE = 0
        served, hits = _run(db, con, DENSE)
    finally:
        PU.KEPT_LINE = line0
    assert served == ['pairusers'] * len(DENSE), served
    assert hits == [0, 0, 0, len(DENSE), 0, 0], hits


def test_pairusers_wide_case_legal():
    """THE WIDE CASE on the dense queries (the dense and kept lines lowered so they stand aside): 1. boxes of 8,
    32, 128, 512 pairs, the sparse line out of reach -- the box grows until the outside bound is met (s x tx is
    480 pairs); 2. a box of 8 then THE SPARSE STEP at once. Every answer legal against DuckDB."""
    db, con, seg = _fixture()
    import wdb_pairusers as PU
    keep = (PU.KEPT_LINE, PU.DENSE_CASE, PU.WIDE_BOX, PU.SPARSE_LINE)
    try:
        PU.KEPT_LINE = 0; PU.DENSE_CASE = 0
        PU.WIDE_BOX = (8, 32, 128, 512); PU.SPARSE_LINE = 1 << 40
        served, hits = _run(db, con, DENSE)
        assert served == ['pairusers'] * len(DENSE), served
        assert hits == [0, 0, 0, 0, len(DENSE), 0], hits
        PU.WIDE_BOX = (8,); PU.SPARSE_LINE = 0
        served, hits = _run(db, con, DENSE)
        assert served == ['pairusers'] * len(DENSE), served
        assert hits == [0, 0, 0, 0, 0, len(DENSE)], hits
    finally:
        PU.KEPT_LINE, PU.DENSE_CASE, PU.WIDE_BOX, PU.SPARSE_LINE = keep
