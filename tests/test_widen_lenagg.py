"""THE EXCLUDED VALUE, ANY (2026-11-07): lenagg's WHERE was exactly `S <> ''`, and '' stood in two roles -- the value the
WHERE leaves out, and the value whose F is 0 (the no-WHERE form's empty rows add F('')). Now `S <> v` for any string v:
v's code and S's NULL code are left out (v absent: nothing but NULL), the no-WHERE form as before. Every answer DuckDB's,
served by lenagg, through every pour: the per-thread boards, the range partition (BIG_KV forced), the grab at the live
rows and the planes of a sparse S whose default is v (the grab's line forced). The old `<> ''` form on a sparse column
whose default is '' reads the planes without decoding S when they are few."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import controller, wdb_lenagg
import test_sparse_default as T

LITS = ("''", "'x'", "'v7'", "'zz'", "'g2'", "'m3'", "'n5'")
COLS = ('s', 'n', 'nn', 'g')


def _forms(col, lit, key):
    w = "WHERE %s <> %s " % (col, lit)
    return ["SELECT %s, AVG(length(%s)) AS a, COUNT(*) AS cn FROM t %sGROUP BY %s ORDER BY %s" % (key, col, w, key, key),
            "SELECT %s, SUM(length(%s)) AS a FROM t %sGROUP BY %s ORDER BY %s" % (key, col, w, key, key),
            "SELECT %s, AVG(length(%s)) AS a, COUNT(*) AS cn FROM t %sGROUP BY %s HAVING COUNT(*) > 1000 "
            "ORDER BY cn DESC LIMIT 7" % (key, col, w, key),
            "SELECT %s, SUM(length(%s)) AS a, COUNT(*) AS cn FROM t %sGROUP BY %s HAVING COUNT(*) <= 1300 "
            "ORDER BY a DESC LIMIT 9" % (key, col, w, key)]


def _check(qs):
    db, con = T._fixture()
    bad = []
    for q in qs:
        r = db.run(q)
        if controller._SERVED[0] != 'lenagg' or not T._legal(q, T._rows(r), con):
            bad.append((controller._SERVED[0], q))
    assert not bad, bad[:5]
    return len(qs)


def _all():
    return [q for col in COLS for lit in LITS for key in ('k', 'c') for q in _forms(col, lit, key)]


def test_any_literal():
    """every column x every literal x key k / c, the boards"""
    return _check(_all())


def test_reversed_and_nulls():
    """`v <> S`; a column with NULLs under '' (NULL rows out of the sums and the counts)"""
    return _check(["SELECT k, AVG(length(nn)) AS a, COUNT(*) AS cn FROM t WHERE 'm3' <> nn GROUP BY k ORDER BY k",
                   "SELECT c, AVG(length(g)) AS a, COUNT(*) AS cn FROM t WHERE '' <> g GROUP BY c ORDER BY c",
                   "SELECT k, COUNT(*) AS cn, SUM(length(g)) AS a FROM t WHERE g <> 'g0' GROUP BY k ORDER BY k"])


def test_no_where_unchanged():
    """the no-WHERE form: served as before (S without NULLs), declined as before (S with NULLs)"""
    n = _check(["SELECT %s, AVG(length(%s)) AS a, COUNT(*) AS cn FROM t GROUP BY %s ORDER BY %s" % (k, c, k, k)
                for c in ('s', 'n') for k in ('k', 'c')])
    db, con = T._fixture()
    for c in ('nn', 'g'):
        q = "SELECT k, AVG(length(%s)) AS a, COUNT(*) AS cn FROM t GROUP BY k ORDER BY k" % c
        try:
            db.run(q)
        except Exception:
            pass                                 # the next read's business
        assert controller._SERVED[0] != 'lenagg', q
    return n


def test_declines():
    """WHERE on another column, a number against text, `=`: declined (another read answers)"""
    db, con = T._fixture()
    for q in ["SELECT k, AVG(length(s)) AS a FROM t WHERE n <> 'n5' GROUP BY k",
              "SELECT k, AVG(length(s)) AS a FROM t WHERE s <> 5 GROUP BY k",
              "SELECT k, AVG(length(s)) AS a FROM t WHERE s = 'x' GROUP BY k"]:
        try:
            db.run(q)
        except Exception:
            pass
        assert controller._SERVED[0] != 'lenagg', q
    return 3


def _forced(**kw):
    old = {k: getattr(wdb_lenagg, k) for k in kw}
    for k, v in kw.items():
        setattr(wdb_lenagg, k, v)
    try:
        return _check([q for col in COLS for lit in ("''", "'x'", "'v7'", "'zz'", "'g2'") for key in ('k', 'c')
                       for q in _forms(col, lit, key)])
    finally:
        for k, v in old.items():
            setattr(wdb_lenagg, k, v)


def test_big_pour():
    """the range partition (every key past BIG_KV)"""
    return _forced(BIG_KV=1)


def test_grab_and_planes():
    """the grab at the live rows and the planes (the line at N: every count of live rows is few)"""
    return _forced(SPARSE_DIV_SMALL=1, SPARSE_DIV=1)


_FX2 = None


def _fixture2():
    """a sparse column whose stored default is '' (e: '' 85%, 'v0'..'v499' scattered) beside keys k (3) and c (300)"""
    global _FX2
    if _FX2 is not None:
        return _FX2
    import duckdb, wdb_encode, pyarrow as pa, pyarrow.parquet as pq
    from wdb_db import Database
    N = 400000
    rng = np.random.default_rng(3)
    i = np.arange(N)
    e = np.full(N, '', object)
    h = rng.random(N) < 0.15
    e[h] = np.char.add('v', rng.integers(0, 500, int(h.sum())).astype(str))
    dd = os.path.join(tempfile.gettempdir(), 'wlg_%s' % uuid.uuid4().hex[:8]); os.makedirs(dd, exist_ok=True)
    try:
        import helpers as _H; _H.register_dir(dd)
    except Exception:
        pass
    p = os.path.join(dd, 't.parquet')
    pq.write_table(pa.table({'k': ((i * 7919) % 3).astype(np.int32), 'c': ((i * 2654435761) % 300).astype(np.int32),
                             'e': pa.array(e.tolist(), pa.string())}), p)
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % p)
    db = Database.create(dd)
    db.cat.add_table('t', [['k', 'int'], ['c', 'int'], ['e', 'str']])
    wdb_encode.encode(p, os.path.join(dd, 't_0.wdb'))
    db.cat.add_segment('t', 't_0.wdb')
    _FX2 = (db, con)
    return _FX2


def test_empty_default_planes():
    """`e <> ''`, e sparse with default '': with the planes few (the line forced) S's codes are never decoded; at the
    measured line the pour, as before. Both DuckDB's."""
    import wdb_policies as P
    db, con = _fixture2()
    segs, decoded = [], []
    o = wdb_lenagg.detect
    wdb_lenagg.detect = lambda s, t, cm: (segs.append(s), o(s, t, cm))[1]
    try:
        qs = ["SELECT %s, AVG(length(e)) AS a, COUNT(*) AS cn FROM t WHERE e <> '' GROUP BY %s ORDER BY %s" % (k, k, k)
              for k in ('k', 'c')] + ["SELECT k, SUM(length(e)) AS a FROM t WHERE e <> 'v7' GROUP BY k ORDER BY k"]
        for div in (None, 1):
            old = wdb_lenagg.SPARSE_DIV_SMALL
            if div is not None:
                wdb_lenagg.SPARSE_DIV_SMALL = div
            try:
                for q in qs:
                    db.run("SELECT k, COUNT(*) FROM t GROUP BY k")
                    r = db.run(q)
                    assert controller._SERVED[0] == 'lenagg', q
                    assert T._legal(q, T._rows(r), con), q
                    seg = segs[-1]
                    assert P.sparse_default_is(seg, 'e', '')
                    if div == 1 and "<> ''" in q:
                        rc = seg._raw_codes          # the planes read never asks for S's codes
                        seg._raw_codes = lambda nm, *a, **kw: (decoded.append(nm), rc(nm, *a, **kw))[1]
                        try:
                            r2 = db.run(q)
                        finally:
                            del seg._raw_codes
                        assert 'e' not in decoded, 'S decoded on the planes path'
                        assert T._rows(r2) == T._rows(r)
            finally:
                wdb_lenagg.SPARSE_DIV_SMALL = old
    finally:
        wdb_lenagg.detect = o
    return 6


if __name__ == '__main__':
    tot = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith('test_') and callable(fn):
            k = fn()
            tot += k or 1
            print('ok', nm, k)
    print('all passed', tot)
