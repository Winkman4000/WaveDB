"""THE REGEX GROUP, GENERAL (2026-10-19, the Q28 board): SELECT REGEXP_REPLACE(S, pat, rep[, 'g']) AS k,
AVG(length(S)), COUNT(*), MIN(S) FROM t [WHERE S <> ''] GROUP BY k HAVING COUNT(*) > n ORDER BY l DESC LIMIT n
is served by regexgroup for a big front-coded text column and a small plain one, a mostly-empty and an
all-empty one, with and without the WHERE, through the prefix-run road and the RE2 road -- equal to DuckDB."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_regexgroup as RG, controller
from wdb_db import Database

TMP = tempfile.gettempdir()
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(28)
    n = 200_001
    hosts = ['h%d.%s' % (i, rng.choice(['com', 'ru', 'org', 'пример'])) for i in range(1500)]
    pre = np.array(['http://', 'https://', 'http://www.', 'https://www.', 'ftp://', ''], dtype=object)
    tail = np.array(['/', '/a/b', '/x?y=1', '/über/', '', '/' + 'p' * 40], dtype=object)
    pool = sorted({pre[rng.integers(0, 6)] + hosts[rng.integers(0, 1500)] + tail[rng.integers(0, 6)]
                   + ('%d' % rng.integers(0, 30) if rng.random() < 0.7 else '') for _ in range(90_000)})
    u = np.array(pool, dtype=object)[rng.integers(0, len(pool), n)]
    u[rng.random(n) < 0.15] = ''
    small = np.array(['', 'http://a.com/', 'https://www.a.com/q', 'http://b.org/x', 'plain', 'Über'],
                     dtype=object)[rng.integers(0, 6, n)]
    sp = np.full(n, '', dtype=object)
    hit = rng.random(n) < 0.01
    sp[hit] = np.array(pool, dtype=object)[rng.integers(0, len(pool), int(hit.sum()))]
    emp = np.full(n, '', dtype=object)
    df = pd.DataFrame({'u': u, 'small': small, 'sp': sp, 'emp': emp})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/rg_{t}.parquet'
    df.to_parquet(pq, index=False)
    d = f'{TMP}/rgdb_{t}'
    db = Database.create(d)
    db.cat.add_table('tbl', [['u', 'string'], ['small', 'string'], ['sp', 'string'], ['emp', 'string']])
    wdb_encode.encode(pq, os.path.join(d, 'tbl_0.wdb'))
    db.cat.add_segment('tbl', 'tbl_0.wdb')
    _FIX = (Database.open(d), pq)
    return _FIX


PATS = [(r"'^https?://(?:www\.)?([^/]+)/.*$'", r"'\1'", ''),        # Q28's own: the prefix class
        (r"'^(\w+)://([^/]*).*$'", r"'\2'", ''),                   # any other pattern: RE2
        (r"'[aeiou]'", "''", ", 'g'")]                              # every match replaced


def _sqls():
    out = []
    for c in ('u', 'small', 'sp', 'emp'):
        for pat, rep, opt in PATS:
            for where in (" WHERE %s <> ''" % c, ''):
                out.append("SELECT REGEXP_REPLACE(%s, %s, %s%s) AS k, AVG(length(%s)) AS l, COUNT(*) AS c, MIN(%s) "
                           "FROM tbl%s GROUP BY k HAVING COUNT(*) > 3 ORDER BY l DESC LIMIT 100000"
                           % (c, pat, rep, opt, c, c, where))
    return out


def _norm(rows):
    return sorted(tuple(round(float(v), 6) if isinstance(v, (float, np.floating)) else
                        (int(v) if isinstance(v, (int, np.integer)) else str(v)) for v in r) for r in rows)


def _duck(pq, sql):
    return duckdb.connect().execute(sql.replace('FROM tbl', "FROM read_parquet('%s')" % pq)).fetchall()


def test_regexgroup_every_column_both_forms():
    db, pq = _fixture()
    roads = RG._RUNROAD, RG._RE2ROAD
    for sql in _sqls():
        controller._SERVED[0] = None
        got = db.run(sql)[0]
        assert controller._SERVED[0] == 'regexgroup', (sql, controller._SERVED[0])
        g, r = _norm(got), _norm(_duck(pq, sql))
        assert g == r, (sql, len(g), len(r), [x for x in g if x not in r][:3], [x for x in r if x not in g][:3])
    assert RG._RE2ROAD > roads[1], 'the RE2 road must have answered'


def test_regexgroup_identity_labels():
    """the identity road (an unmatched value is its own label) and the copying road give DuckDB's answers;
    the fixture holds bare hosts ('h12.ru') equal to captures of 'http://h12.ru/...', so the byte check runs"""
    db, pq = _fixture()
    sqls = [s for s in _sqls() if s.startswith('SELECT REGEXP_REPLACE(u, ' + PATS[0][0])]
    old = RG.IDENTITY
    try:
        for on in (True, False):
            RG.IDENTITY = on
            n0 = RG._IDROAD
            for seg in _segs(db):
                seg.__dict__.pop('_rg_memo', None)
            for sql in sqls:
                got = db.run(sql)[0]
                assert _norm(got) == _norm(_duck(pq, sql)), (on, sql)
            assert (RG._IDROAD > n0) == on, (on, RG._IDROAD, n0)
    finally:
        RG.IDENTITY = old


def _segs(db):
    import gc, wdb_engine
    return [o for o in gc.get_objects() if isinstance(o, wdb_engine.Segment)]


def test_regexgroup_limit_without_having():
    """no HAVING: every group survives, and the LIMIT cut is taken before any label becomes a string. A tie
    at the LIMIT is legal when every row is a real group and the ordering column carries DuckDB's values."""
    db, pq = _fixture()
    for c in ('u', 'small', 'sp'):
        for pat, rep, opt in PATS[:2]:
            for oc, oi in (('c', 2), ('l', 1), ('m', 3)):
                sql = ("SELECT REGEXP_REPLACE(%s, %s, %s%s) AS k, AVG(length(%s)) AS l, COUNT(*) AS c, MIN(%s) AS m "
                       "FROM tbl GROUP BY k ORDER BY %s DESC LIMIT 7" % (c, pat, rep, opt, c, c, oc))
                controller._SERVED[0] = None
                got = db.run(sql)[0]
                assert controller._SERVED[0] == 'regexgroup', (sql, controller._SERVED[0])
                ref = _duck(pq, sql)
                full = set(_norm(_duck(pq, sql.rsplit(' LIMIT', 1)[0])))
                assert len(got) == len(ref) and set(_norm(got)) <= full, (sql, got[:2], ref[:2])
                key = (lambda v: round(float(v), 6)) if oi < 3 else str
                assert [key(r[oi]) for r in got] == [key(r[oi]) for r in ref], (sql, got, ref)
