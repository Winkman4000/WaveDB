"""THE REGEX GROUP, ANY LITERAL (2026-11-07, the widening sweep): regexgroup's WHERE was `col <> ''` only. Now
`col <> 'lit'` for any text literal (either side): lit's dictionary code carries no weight (a literal the column never
holds: none), NULL rows never count. HAVING takes COUNT(*) with any comparison against an integer, ORDER BY any one
projection by alias, ordinal or its own expression, ASC or DESC. GROUP BY must name the regex key (GROUP BY col keeps
the values apart; a GROUP BY name that is also a table column is that column). Every answer DuckDB's (ties at a
LIMIT cut legal); the shapes the read cannot be exact on decline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import test_sparse_default as T

PATS = ["'^v(.)'", "'^(.)'"]
LITS = ["''", "'x'", "'v7'", "'zz'"]
EXTRA = {'g': ["'g2'"], 'nn': ["'m3'"], 'n': ["'n7'"]}


def _q(col, pat, where, tail="ORDER BY cc DESC LIMIT 10"):
    return ("SELECT REGEXP_REPLACE(%s, %s, '\\1') AS rk, AVG(length(%s)) AS l, COUNT(*) AS cc, MIN(%s) AS mn "
            "FROM t%s GROUP BY rk %s" % (col, pat, col, col, where, tail))


def _seg():
    db, _ = T._fixture()
    return db.open_segment(db.cat.segment_paths('t')[0], 't')


def _detect(q):
    import wdb_regexgroup as RG, wdb_db
    return RG.detect(_seg(), wdb_db._parse_sql_cached(q), {})


def _legal(q, got, con, oi):
    """exact, or a LIMIT cut with ties: column oi (the ORDER BY's) exact, every row a true group"""
    want = T._rows(con.execute(q).fetchall())
    if got == want:
        return True
    if oi is None or len(got) != len(want) or [x[oi] for x in got] != [x[oi] for x in want]:
        return False
    full = set(T._rows(con.execute(q[:q.index(' ORDER BY ')]).fetchall()))
    return all(x in full for x in got)


def _check(qs, served=True):
    """each q, or (q, the ORDER BY's column index) for a cut whose ties DuckDB may break otherwise"""
    import controller
    db, con = T._fixture()
    bad = []
    for q in qs:
        q, oi = q if isinstance(q, tuple) else (q, 2)
        got = T._rows(db.run(q))
        if served and controller._SERVED[0] != 'regexgroup':
            bad.append(('not served: %s' % controller._SERVED[0], q))
        elif not _legal(q, got, con, oi):
            bad.append(('wrong', q, got[:4], T._rows(con.execute(q).fetchall())[:4]))
    assert not bad, bad


def test_any_literal_is_duckdbs():
    qs = []
    for col in ('s', 'n', 'nn', 'g'):
        for pat in PATS:
            for lit in LITS + EXTRA.get(col, []):
                qs.append(_q(col, pat, " WHERE %s <> %s" % (col, lit)))
    qs.append(_q('s', PATS[0], " WHERE 'x' <> s"))                      # the literal on the left
    _check(qs)


def test_open_form_still_served_and_nulls_still_declined():
    _check([_q(c, p, '') for c in ('s', 'n') for p in PATS])
    for c in ('nn', 'g'):
        assert _detect(_q(c, PATS[1], '')) is None                    # the open form keeps NULL a group: declined


def test_having_any_count_comparison():
    qs = []
    for op in ('>', '>=', '<', '<=', '=', '<>'):
        for n in (59, 20000):
            qs.append(_q('s', PATS[0], " WHERE s <> 'x'", "HAVING COUNT(*) %s %d ORDER BY cc DESC LIMIT 10" % (op, n)))
    qs.append(_q('s', PATS[0], " WHERE s <> 'x'", "HAVING 100 < COUNT(*) ORDER BY cc DESC LIMIT 10"))
    qs.append(_q('g', PATS[1], " WHERE g <> 'g2'", "HAVING COUNT(*) >= 80037 ORDER BY cc DESC LIMIT 10"))
    _check(qs)


def test_order_by_ordinal_expression_and_asc():
    _check([_q('s', PATS[0], " WHERE s <> 'v7'", "ORDER BY 3 DESC LIMIT 10"),
            _q('s', PATS[0], " WHERE s <> 'v7'", "ORDER BY COUNT(*) DESC LIMIT 10"),
            (_q('s', PATS[0], " WHERE s <> 'v7'", "ORDER BY rk LIMIT 10"), 0),
            (_q('s', PATS[0], " WHERE s <> 'v7'", "ORDER BY rk DESC LIMIT 10"), 0),
            (_q('s', PATS[0], " WHERE s <> 'v7'", "ORDER BY 1 LIMIT 10"), 0),
            (_q('s', PATS[0], " WHERE s <> 'v7'", "ORDER BY mn LIMIT 10"), 3),
            (_q('s', PATS[0], " WHERE s <> 'v7'", "ORDER BY MIN(s) DESC LIMIT 10"), 3),
            (_q('nn', PATS[1], " WHERE nn <> 'm3'", "ORDER BY l LIMIT 10"), 1),
            _q('g', PATS[1], " WHERE g <> 'zz'", "ORDER BY cc LIMIT 10"),
            _q('s', PATS[0], " WHERE s <> 'x'", "ORDER BY cc LIMIT 10"),         # an ASC cut through ties
            (_q('n', PATS[1], " WHERE n <> 'n7'", "ORDER BY rk"), 0)])


def test_group_by_ordinal_and_expression():
    _check(["SELECT REGEXP_REPLACE(s, '^(.).*$', '\\1') AS rk, COUNT(*) AS cc FROM t WHERE s <> 'v7' GROUP BY 1 "
            "ORDER BY cc DESC",
            "SELECT REGEXP_REPLACE(s, '^(.).*$', '\\1') AS rk, COUNT(*) AS cc FROM t WHERE s <> '' "
            "GROUP BY REGEXP_REPLACE(s, '^(.).*$', '\\1') ORDER BY cc DESC"])


def test_declines():
    # the groups are col's values, not the labels; a GROUP BY name that is a table column; a number literal;
    # another column's filter; a number column; two ORDER BY keys; HAVING on another aggregate
    for q in ["SELECT REGEXP_REPLACE(s, '^(.).*$', '\\1') AS rk, COUNT(*) AS cc FROM t GROUP BY s ORDER BY cc DESC",
              "SELECT REGEXP_REPLACE(s, '^(.).*$', '\\1') AS k, COUNT(*) AS cc FROM t GROUP BY k",
              _q('s', PATS[0], " WHERE s <> 7"),
              _q('s', PATS[0], " WHERE n <> 'n7'"),
              "SELECT REGEXP_REPLACE(k, '^(.)', '\\1') AS rk, COUNT(*) AS cc FROM t GROUP BY rk",
              _q('s', PATS[0], '', "ORDER BY cc DESC, rk LIMIT 10"),
              _q('s', PATS[0], '', "HAVING AVG(length(s)) > 1 ORDER BY cc DESC LIMIT 10"),
              _q('s', PATS[0], '', "HAVING COUNT(*) > 1.5 ORDER BY cc DESC LIMIT 10")]:
        assert _detect(q) is None, q


def test_any_literal_costs_what_the_old_one_did():
    """`<> ''` and every other literal take the same road: one per-code count (the census or one bincount), no row
    filter -- the literal only picks which code's weight is taken away"""
    import wdb_regexgroup as RG, wdb_wherescan as WS
    calls = []
    def counted(seg, col):
        calls.append(col)
        return saved[0](seg, col)
    def boom(*a, **k):
        raise AssertionError('a row filter')
    saved = RG._code_counts, WS.neq_rows
    RG._code_counts, WS.neq_rows = counted, boom
    try:
        for lit in LITS:
            del calls[:]
            _check([_q('s', PATS[0], " WHERE s <> %s" % lit)])
            assert calls == ['s'], (lit, calls)
    finally:
        RG._code_counts, WS.neq_rows = saved


if __name__ == '__main__':
    for nm, fn in list(globals().items()):
        if nm.startswith('test_'):
            fn(); print('PASS', nm)
