"""A SCALAR OF THE GROUP KEYS (2026-11-07): the general GROUP BY emitted the raw key for a projection that is a function
of it -- `SELECT length(s) ... GROUP BY s` gave 'x' where DuckDB gives 1 (upper, ||, REGEXP_REPLACE, EXTRACT of a grouped
time column alike). And a scalar function of a column with NULLs, as a key or under an aggregate, raised IndexError (the
per-code table had no entry for NULL's code). Every answer DuckDB's (ties at a LIMIT cut legal), on the general path
itself (wdb_sql.execute) and through the router."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

_SEG = [None]


def _fx():
    import test_sparse_default as T
    db, con = T._fixture()
    if _SEG[0] is None:
        from wdb_engine import Segment
        _SEG[0] = Segment(db.cat.segment_paths('t')[0])
    return T, db, con, _SEG[0]


def _check(qs, routed=True):
    import wdb_sql
    T, db, con, seg = _fx()
    bad = []
    for q in qs:
        got = T._rows(wdb_sql.execute(seg, q))                 # the general path itself
        if not T._legal(q, got, con):
            bad.append(('general', q, got[:4], T._rows(con.execute(q).fetchall())[:4]))
        if routed:
            got = T._rows(db.run(q))                          # whichever read serves it
            if not T._legal(q, got, con):
                bad.append(('routed', q, got[:4], T._rows(con.execute(q).fetchall())[:4]))
    assert not bad, '\n'.join(map(str, bad))


def test_text_key_functions():
    _check(["SELECT length(s) AS L, COUNT(*) AS c FROM t GROUP BY s ORDER BY c DESC LIMIT 5",
            "SELECT upper(s) AS U, COUNT(*) AS c FROM t GROUP BY s ORDER BY c DESC LIMIT 5",
            "SELECT lower(upper(n)) AS lo, COUNT(*) AS c FROM t GROUP BY n ORDER BY c DESC LIMIT 5",
            "SELECT s || 'z' AS sz, COUNT(*) AS c FROM t GROUP BY s ORDER BY c DESC LIMIT 5",
            "SELECT concat(s, '-', s) AS cc, COUNT(*) AS c FROM t GROUP BY s ORDER BY c DESC LIMIT 5",
            "SELECT substring(n, 2, 1) AS sb, s, COUNT(*) AS c FROM t GROUP BY n, s ORDER BY c DESC LIMIT 5",
            "SELECT s, length(s) AS L, COUNT(*) AS c FROM t GROUP BY s ORDER BY c DESC LIMIT 5"])


def test_regexp_replace_of_key():
    # served by regexgroup in this tree through the router; the general path must be right on its own
    _check(["SELECT REGEXP_REPLACE(s, '^(.).*$', '\\1') AS rk, COUNT(*) AS c FROM t GROUP BY s ORDER BY c DESC LIMIT 5",
            "SELECT REGEXP_REPLACE(nn, '^(.).*$', '\\1') AS rk, COUNT(*) AS c FROM t GROUP BY nn ORDER BY c DESC LIMIT 5",
            "SELECT REGEXP_REPLACE(g, 'g', 'h') AS rk, COUNT(*) AS c FROM t GROUP BY REGEXP_REPLACE(g, 'g', 'h') ORDER BY c DESC LIMIT 5"],
           routed=False)


def test_int_key_functions():
    _check(["SELECT abs(k - 1) AS a, COUNT(*) AS c FROM t GROUP BY k ORDER BY c DESC LIMIT 5",
            "SELECT k * 2 AS k2, COUNT(*) AS c FROM t GROUP BY k ORDER BY c DESC LIMIT 5",
            "SELECT k % 2 AS km, COUNT(*) AS c FROM t GROUP BY k ORDER BY c DESC LIMIT 5",
            "SELECT k + 1 AS k1, COUNT(*) AS c FROM t GROUP BY k ORDER BY c DESC LIMIT 5",
            "SELECT c % 7 AS cm, SUM(f) AS sf, COUNT(*) AS cnt FROM t GROUP BY c ORDER BY cnt DESC LIMIT 5"])


def test_two_keys():
    _check(["SELECT length(s) AS L, k % 2 AS km, COUNT(*) AS c FROM t GROUP BY s, k ORDER BY c DESC LIMIT 5",
            "SELECT k * 10 AS kk, upper(g) AS ug, COUNT(*) AS c FROM t GROUP BY k, g ORDER BY c DESC LIMIT 6",
            "SELECT s || g AS sg, COUNT(*) AS c FROM t GROUP BY s, g ORDER BY c DESC LIMIT 5"])


def test_time_key_parts():
    _check(["SELECT u, extract(minute FROM tm) AS m, s, COUNT(*) AS c FROM t GROUP BY u, tm, s ORDER BY c DESC LIMIT 10",
            "SELECT extract(hour FROM tm) AS h, COUNT(*) AS c FROM t GROUP BY tm ORDER BY c DESC LIMIT 5",
            "SELECT DATE_TRUNC('minute', tm) AS tr, COUNT(*) AS c FROM t GROUP BY tm ORDER BY c DESC LIMIT 5",
            "SELECT extract(day FROM d) AS dd, COUNT(*) AS c FROM t GROUP BY d ORDER BY c DESC LIMIT 5"])


def test_having_and_order():
    _check(["SELECT length(s) AS L, COUNT(*) AS c FROM t GROUP BY s HAVING COUNT(*) > 90 ORDER BY c DESC LIMIT 5",
            "SELECT upper(s) AS U, COUNT(*) AS c FROM t GROUP BY s HAVING COUNT(*) > 85 ORDER BY U LIMIT 10",
            "SELECT upper(nn) AS U, COUNT(*) AS c FROM t GROUP BY nn ORDER BY U LIMIT 10",
            "SELECT k % 2 AS km, k, COUNT(*) AS c FROM t GROUP BY k ORDER BY km, k",
            "SELECT length(s) AS L, AVG(k) AS a, COUNT(*) AS c FROM t GROUP BY s HAVING COUNT(*) > 88 ORDER BY c DESC LIMIT 5"])


def test_nulls():
    _check(["SELECT length(nn) AS L, COUNT(*) AS c FROM t GROUP BY nn ORDER BY c DESC LIMIT 10",
            "SELECT nn || 'z' AS z, COUNT(*) AS c FROM t GROUP BY nn ORDER BY c DESC LIMIT 10",
            "SELECT coalesce(g, 'none') AS z, COUNT(*) AS c FROM t GROUP BY g ORDER BY c DESC LIMIT 10",
            "SELECT length(nn) AS L, COUNT(*) AS c FROM t GROUP BY length(nn) ORDER BY c DESC LIMIT 10",
            "SELECT length(g) AS L, COUNT(*) AS c FROM t GROUP BY length(g) ORDER BY c DESC LIMIT 10",
            "SELECT g, AVG(length(nn)) AS a, COUNT(*) AS c FROM t GROUP BY g ORDER BY c DESC LIMIT 10",
            "SELECT k, SUM(length(g)) AS sl, MAX(length(nn)) AS mx, COUNT(*) AS c FROM t GROUP BY k ORDER BY c DESC LIMIT 5",
            "SELECT length(g) AS L, AVG(length(nn)) AS a, COUNT(*) AS c FROM t GROUP BY length(g) ORDER BY c DESC LIMIT 5",
            "SELECT REGEXP_REPLACE(nn, '^(.).*$', '\\1') AS rk, COUNT(*) AS c FROM t GROUP BY REGEXP_REPLACE(nn, '^(.).*$', '\\1') ORDER BY c DESC LIMIT 5"])


def test_not_a_key_raises():
    import wdb_sql
    T, db, con, seg = _fx()
    for q in ["SELECT length(u) AS L, COUNT(*) AS c FROM t GROUP BY s",                 # u is not a key
              "SELECT length(s) + 1 AS L, COUNT(*) AS c FROM t GROUP BY length(s)"]:   # s itself is not a key
        try:
            r = wdb_sql.execute(seg, q)
        except NotImplementedError:
            continue
        raise AssertionError('emitted %r for %s' % (r[0][:2], q))


if __name__ == '__main__':
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    ok = 0
    for f in fns:
        try:
            f(); ok += 1; print('PASS', f.__name__)
        except Exception:
            print('FAIL', f.__name__); traceback.print_exc()
    print('%d/%d passed' % (ok, len(fns)))
    sys.exit(0 if ok == len(fns) else 1)
