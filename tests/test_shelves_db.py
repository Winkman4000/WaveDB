"""THE SHELVES (tag 19 labels by code range), end to end through the Database: loaded by the kit's
own path (streaming encode, column jobs in worker processes), queried through Database.run (the
suite runs WDB_QMEM_STRICT: nothing data-derived may outlive a query), with NULLs in the column,
then INSERT / DELETE (segment mode re-encodes; buffered mode keeps the cold segment and merges a
hot buffer). Every answer against DuckDB. Also: the shelf count's own rule and the table span."""
import sys, os, uuid, tempfile, shutil, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode
from wdb_db import Database
from wdb_catalog import Catalog
from wdb_engine import Segment

TMP = tempfile.gettempdir()


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _frame(n=260_000, seed=28):
    """users in sessions (each 16K-row stretch sees a few thousand of 90K users), ~3% NULL users"""
    rng = np.random.default_rng(seed)
    uid = np.empty(n, np.int64)
    for b in range(0, n, 16384):
        live = rng.integers(0, 90_000, 2500)
        uid[b:b + 16384] = live[rng.integers(0, 2500, min(16384, n - b))]
    user = np.array([f'u{u:05d}' for u in uid], dtype=object)
    user[rng.random(n) < 0.03] = None
    return pd.DataFrame({'user': user, 'uid': uid * 7919 + 13, 'k': rng.integers(0, 30, n).astype(np.int64)})


def _load(df, R=7):
    """the kit's load (bin/wdb cmd_load): streaming encode into a realm, then the catalog entry"""
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'shdb_{t}'); pq = os.path.join(TMP, f'shdb_{t}.parquet')
    df.to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    out = os.path.join(d, 'hits_0.wdb')
    with _env(WDB_E19_FORCE='1', WDB_E19_SHELVES=R):
        wdb_encode.encode(pq, out, stream=True)
    seg = Segment(out)
    cat = Catalog.open(d)
    tn = {0: 'int', 1: 'str', 2: 'float'}
    cat.data['tables']['hits'] = {'schema': [[c, tn.get(seg.cols[c].get('dt'), 'str')] for c in seg.order],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    return d, pq, seg


def _norm(rows):
    out = []
    for r in rows:
        out.append(tuple(x.decode() if isinstance(x, bytes) else
                         (round(float(x), 4) if isinstance(x, (int, float)) and not isinstance(x, bool) else x)
                         for x in r))
    return sorted(out, key=lambda t: tuple(str(x) for x in t))


def _queries(df):
    u = df['user'].dropna()
    a = u.iloc[len(u) // 200]; b = u.iloc[len(u) * 3 // 4]
    v = int(df['uid'].iloc[len(df) * 3 // 10])
    return [f"SELECT COUNT(*) FROM hits WHERE user = '{a}'",
            f"SELECT k, COUNT(*) FROM hits WHERE user = '{a}' GROUP BY k",
            f"SELECT user, k FROM hits WHERE user = '{b}' ORDER BY k, user LIMIT 5",
            f"SELECT COUNT(*) FROM hits WHERE user IN ('{a}', '{b}', 'u99999x')",
            f"SELECT COUNT(*) FROM hits WHERE user <> '{a}'",
            "SELECT COUNT(*) FROM hits WHERE user = 'no-such-user'",
            "SELECT COUNT(*) FROM hits WHERE user IS NULL",
            "SELECT COUNT(user), COUNT(*) FROM hits",
            "SELECT user, COUNT(*) AS c FROM hits GROUP BY user ORDER BY c DESC, user LIMIT 10",
            "SELECT COUNT(DISTINCT user) FROM hits",
            "SELECT k, COUNT(DISTINCT user) FROM hits GROUP BY k ORDER BY k LIMIT 8",
            "SELECT MIN(user), MAX(user) FROM hits WHERE k = 3",
            f"SELECT COUNT(*) FROM hits WHERE uid = {v}",
            "SELECT user, uid FROM hits WHERE k = 7 ORDER BY uid, user LIMIT 12"]


def _duck(src, sql):
    con = duckdb.connect()
    if isinstance(src, pd.DataFrame):
        con.register('hits_df', src)
        return _norm(con.execute(sql.replace('FROM hits', 'FROM hits_df')).fetchall())
    return _norm(con.execute(sql.replace('FROM hits', f"FROM '{src}'")).fetchall())


def test_shelf_count_rule():
    """under 2 shelves' worth of labels: kept by block; ~256 KB a shelf; at most 4096; the switch"""
    B = wdb_encode.E19_SHELF_BYTES
    with _env(WDB_E19_SHELVES='auto'):
        assert wdb_encode._e19_shelf_count(0) == 0
        assert wdb_encode._e19_shelf_count(2 * B - 1) == 0
        assert wdb_encode._e19_shelf_count(2 * B) == 2
        assert wdb_encode._e19_shelf_count(221 * B + 5) == 221
        assert wdb_encode._e19_shelf_count(10_000 * B) == 4096
    with _env(WDB_E19_SHELVES='0'):
        assert wdb_encode._e19_shelf_count(10_000 * B) == 0
    with _env(WDB_E19_SHELVES='1'):
        assert wdb_encode._e19_shelf_count(0) == 2
    with _env(WDB_E19_SHELVES='9'):
        assert wdb_encode._e19_shelf_count(0) == 9


def test_stream_loaded_shelves_answer_like_duck():
    """the kit's load path makes the shelved column; every query twice (the second one warm), both
    equal to DuckDB, through Database.run under the strict law; the table span covers the tables"""
    df = _frame()
    d, pq, seg = _load(df)
    try:
        c = seg.cols['user']
        assert c['code_enc'] == 19 and c.get('e19R') == 7 and 'e19doff' not in c, (c.get('code_enc'), c.get('e19R'))
        lo, hi = c['e19tab']                                  # the span warm_span reads: SW, pre, soff
        base = seg.buf.ctypes.data
        for arr in (c['e19SW'], c['e19pre'], c['e19soff']):
            a0 = arr.ctypes.data - base
            assert lo <= a0 and a0 + arr.nbytes <= hi
        assert c['e19SW'].ctypes.data - base == lo and c['e19soff'].ctypes.data - base + c['e19soff'].nbytes == hi
        db = Database.open(d)
        for sql in _queries(df):
            want = _duck(pq, sql)
            for rep in range(2):
                got = _norm(db.run(sql)[0])
                assert got == want, (rep, sql, got[:3], want[:3])
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_shelved_table_after_insert_and_delete():
    """segment mode: INSERT (a new user and old ones) and DELETE re-encode the table -- the column stays
    shelved and every answer equals DuckDB's over the same rows"""
    df = _frame(n=120_000, seed=5)
    d, pq, seg = _load(df)
    try:
        db = Database.open(d)
        a = df['user'].dropna().iloc[10]
        new = pd.DataFrame({'user': [a, 'zz-new', None, a], 'uid': [1, 2, 3, 4], 'k': [99, 98, 97, 96]})
        order = [c for c, t in db.cat.get_table('hits')['schema']]      # VALUES follow the table's order
        lit = lambda v: ("NULL" if v is None or (isinstance(v, float) and v != v)
                         else ("'%s'" % v if isinstance(v, str) else str(int(v))))
        with _env(WDB_E19_FORCE='1', WDB_E19_SHELVES=7):
            db.run("INSERT INTO hits VALUES " + ",".join(
                "(" + ", ".join(lit(r[c]) for c in order) + ")" for r in new.to_dict('records')))
            n = db.run("DELETE FROM hits WHERE k = 3")
        sp = db.cat.segment_paths('hits')
        assert len(sp) == 1 and Segment(sp[0]).cols['user'].get('e19R') == 7   # re-encoded, still shelved
        cur = pd.concat([df, new], ignore_index=True)
        assert n == int((cur['k'] == 3).sum()), n
        cur = cur[cur['k'] != 3].reset_index(drop=True)
        for sql in _queries(cur) + ["SELECT COUNT(*) FROM hits WHERE user = 'zz-new'",
                                    f"SELECT uid, k FROM hits WHERE user = '{a}' AND k > 90"]:
            assert _norm(db.run(sql)[0]) == _duck(cur, sql), sql
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
