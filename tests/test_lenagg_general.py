"""THE LENGTH POUR, GENERAL (2026-10-18, the Q27 board): AVG/SUM(length(S)) [+ COUNT(*)] GROUP BY K
[WHERE S <> ''] [HAVING COUNT(*) op n] ORDER BY <projection> [ASC|DESC] LIMIT n is served by the registered read
lenagg for keys of any stored encoding (a staircase key, a random key, a string key), with and without stored
row lengths, through both pours (per-thread boards, and the range partition past BIG_KV) -- equal to DuckDB."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_lens, wdb_lenagg, controller
from wdb_db import Database

TMP = tempfile.gettempdir()
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(27)
    n = 300_001
    alpha = list('abcdefghijklmnopqrstuvwxyz/._-') + ['é', '日', 'ж']
    pool = sorted(set(''.join(rng.choice(alpha, int(m))) for m in rng.integers(1, 120, 60000)))   # front-coded
    s = np.array(pool, dtype=object)[rng.integers(0, len(pool), n)]
    s[rng.random(n) < 0.2] = ''
    tag = np.array(['', 'x', 'yy', 'zzz', 'wwww'], dtype=object)[rng.integers(0, 5, n)]   # a small text column
    sp = np.full(n, '', dtype=object)                                    # mostly empty: the grab's case
    hit = rng.random(n) < 0.004
    sp[hit] = np.array(pool, dtype=object)[rng.integers(0, len(pool), int(hit.sum()))]
    emp = np.full(n, '', dtype=object)                                   # every row empty: no live row
    df = pd.DataFrame({
        'st': np.sort(rng.integers(0, 30, n)).astype(np.int64),          # sorted: a staircase key
        'rk': rng.integers(0, 2000, n).astype(np.int64),                 # random key
        'wide': rng.integers(0, 150_000, n).astype(np.int64),            # many keys (the big pour, forced)
        'sk': np.array(['k%02d' % i for i in range(40)], dtype=object)[rng.integers(0, 40, n)],   # string key
        's': s, 'tag': tag, 'sp': sp, 'emp': emp})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/lg_{t}.parquet'
    df.to_parquet(pq, index=False)
    d = f'{TMP}/lgdb_{t}'
    db = Database.create(d)
    db.cat.add_table('tbl', [['st', 'int'], ['rk', 'int'], ['wide', 'int'], ['sk', 'string'], ['s', 'string'],
                             ['tag', 'string'], ['sp', 'string'], ['emp', 'string']])
    keep = {k: os.environ.get(k) for k in ('WDB_ROWLEN_COLS', 'WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    os.environ.update(WDB_ROWLEN_COLS='s', WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0')
    try:
        wdb_encode.encode(pq, os.path.join(d, 'tbl_0.wdb'))
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    db.cat.add_segment('tbl', 'tbl_0.wdb')
    _FIX = (Database.open(d), pq, os.path.join(d, 'tbl_0.wdb'))
    return _FIX


def _sqls():
    out = []
    for k in ('st', 'rk', 'sk', 'wide'):
        for sc in ('s', 'tag', 'sp', 'emp'):
            out += [
                "SELECT %s, AVG(length(%s)) AS l, COUNT(*) AS c FROM tbl WHERE %s <> '' GROUP BY %s "
                "HAVING COUNT(*) > 50 ORDER BY l DESC LIMIT 25" % (k, sc, sc, k),
                "SELECT %s, AVG(length(%s)) AS l, COUNT(*) AS c FROM tbl GROUP BY %s ORDER BY c DESC LIMIT 9" % (k, sc, k),
                "SELECT %s, SUM(length(%s)) AS l FROM tbl WHERE %s <> '' GROUP BY %s ORDER BY %s ASC LIMIT 12"
                % (k, sc, sc, k, k),
                "SELECT %s, AVG(length(%s)) AS l, COUNT(*) AS c FROM tbl GROUP BY %s HAVING COUNT(*) <= 9000 "
                "ORDER BY l ASC LIMIT 6" % (k, sc, k)]
    return out


def _norm(rows):
    return sorted(tuple(round(float(v), 6) if isinstance(v, (float, np.floating)) else
                        (int(v) if isinstance(v, (int, np.integer)) else str(v)) for v in r) for r in rows)


def _duck(pq, sql):
    return duckdb.connect().execute(sql.replace('FROM tbl', "FROM read_parquet('%s')" % pq)).fetchall()


def _check(db, pq, sqls):
    for sql in sqls:
        controller._SERVED[0] = None
        got, hdrs = db.run(sql)[:2]
        ref = _duck(pq, sql)
        assert controller._SERVED[0] == 'lenagg', (sql, controller._SERVED[0])
        # an ORDER BY can tie at the LIMIT: the answer is legal when every returned row is a real group with
        # DuckDB's values and the ordering column carries DuckDB's values, in DuckDB's order
        g, r = _norm(got), _norm(ref)
        if g != r:
            full = {x for x in _norm(_duck(pq, sql.rsplit(' LIMIT', 1)[0]))}
            oi = list(hdrs).index(sql.split('ORDER BY ')[1].split()[0])
            assert set(g) <= full and len(g) == len(r), (sql, g[:3], r[:3])
            assert [round(float(x[oi]), 6) if not isinstance(x[oi], str) else x[oi] for x in got] == \
                   [round(float(x[oi]), 6) if not isinstance(x[oi], str) else x[oi] for x in ref], sql


def test_lenagg_every_key_both_pours():
    db, pq, w = _fixture()
    assert os.path.exists(wdb_lens.row_path(w, 's')), 'the fixture must carry stored row lengths for s'
    _check(db, pq, _sqls())
    old = wdb_lenagg.BIG_KV
    try:
        wdb_lenagg.BIG_KV = 10                       # every key through the range partition
        _check(db, pq, _sqls())
    finally:
        wdb_lenagg.BIG_KV = old
    old = wdb_lenagg.SPARSE_DIV, wdb_lenagg.SPARSE_DIV_SMALL
    try:
        for div in (1, 0):                           # every filtered query grouped at its live rows; then never
            wdb_lenagg.SPARSE_DIV = wdb_lenagg.SPARSE_DIV_SMALL = div
            _check(db, pq, _sqls())
    finally:
        wdb_lenagg.SPARSE_DIV, wdb_lenagg.SPARSE_DIV_SMALL = old


def test_lenagg_without_row_lengths():
    db, pq, w = _fixture()
    p = wdb_lens.row_path(w, 's'); moved = p + '.off'
    os.replace(p, moved)
    try:
        _check(Database.open(os.path.dirname(w)), pq, _sqls())
    finally:
        os.replace(moved, p)


def test_lenagg_live_rows_first():
    """THE LIVE ROWS FIRST: the line opened to every encoding and every row count (LIVE_SHIFT 0), so the column with
    stored row lengths (s: 20% empty) is read at its non-empty rows only -- filtered (grouped there) and open (sums
    there, counts from the key's census) -- for every key, equal to DuckDB (or a legal tie); the columns without
    stored row lengths (sp, emp) keep the current read."""
    db, pq, w = _fixture()
    old = wdb_lenagg.LIVE_ENC, wdb_lenagg.LIVE_SHIFT
    try:
        wdb_lenagg.LIVE_ENC = tuple(range(64)); wdb_lenagg.LIVE_SHIFT = 0
        qs = [q for q in _sqls() if 'length(s)' in q]
        h0 = wdb_lenagg._LIVE_HITS[0]
        _check(db, pq, qs)
        assert wdb_lenagg._LIVE_HITS[0] - h0 == len(qs), (wdb_lenagg._LIVE_HITS[0] - h0, len(qs))
        qs = [q for q in _sqls() if 'length(sp)' in q or 'length(emp)' in q]
        h0 = wdb_lenagg._LIVE_HITS[0]
        _check(db, pq, qs)
        assert wdb_lenagg._LIVE_HITS[0] - h0 == 0, 'no stored row lengths for sp / emp: the line keeps them out'
    finally:
        wdb_lenagg.LIVE_ENC, wdb_lenagg.LIVE_SHIFT = old


def test_lenagg_too_few_rows():
    """THE TOO-FEW-ROWS BOUND: under WHERE S <> '' a group holds at most k rows (k = the kept rows: sp ~1,200 of
    300,001, s ~240,000, emp none). A HAVING no count from 1 to k passes is answered empty without reading the key;
    one some count passes is read as before. Both through the grab (every filtered query grouped at its rows) and the
    live rows first (s read at its non-empty rows), for every key, equal to DuckDB."""
    db, pq, w = _fixture()
    q = ("SELECT %s, AVG(length(%s)) AS l, COUNT(*) AS c FROM tbl WHERE %s <> '' GROUP BY %s "
         "HAVING COUNT(*) %s ORDER BY l DESC LIMIT 25")
    keys = ('st', 'rk', 'sk', 'wide')
    fire = [q % (k, sc, sc, k, h) for k in keys for sc, h in (
        ('sp', '> 5000'), ('sp', '>= 5001'), ('sp', '= 4000'), ('s', '> 300001'), ('emp', '< 3'), ('emp', '> 0'))]
    keep = [q % (k, sc, sc, k, h) for k in keys for sc, h in (
        ('sp', '> 1'), ('sp', '< 3'), ('sp', '= 1'), ('sp', '<= 2'), ('s', '> 50'), ('s', '>= 1'))]
    old = wdb_lenagg.SPARSE_DIV, wdb_lenagg.SPARSE_DIV_SMALL, wdb_lenagg.LIVE_ENC, wdb_lenagg.LIVE_SHIFT
    try:
        wdb_lenagg.SPARSE_DIV = wdb_lenagg.SPARSE_DIV_SMALL = 1
        for live in (False, True):
            if live:
                wdb_lenagg.LIVE_ENC = tuple(range(64)); wdb_lenagg.LIVE_SHIFT = 0
            h0 = wdb_lenagg._BOUND_HITS[0]
            _check(db, pq, fire)
            assert wdb_lenagg._BOUND_HITS[0] - h0 == len(fire), (live, wdb_lenagg._BOUND_HITS[0] - h0, len(fire))
            h0 = wdb_lenagg._BOUND_HITS[0]
            _check(db, pq, keep)
            assert wdb_lenagg._BOUND_HITS[0] - h0 == 0, (live, 'a count from 1 to k passes: the key must be read')
    finally:
        (wdb_lenagg.SPARSE_DIV, wdb_lenagg.SPARSE_DIV_SMALL, wdb_lenagg.LIVE_ENC, wdb_lenagg.LIVE_SHIFT) = old
