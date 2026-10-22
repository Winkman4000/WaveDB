"""THE CURSOR (2026-10-22): db.execute(sql) -> wdb_result.Cursor, PEP 249's description / rowcount / fetchone /
fetchmany / fetchall / iteration plus fetchnumpy, arrow, fetch_record_batch and df (DuckDB's and ClickHouse's Python
client forms). A read that can answer in columns does (groupself's bulk path: text decoded per fetched batch); every
other answer is wrapped as run() returns it. Oracle: db.run and DuckDB."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import sqlglot
import wdb_result as WR, wdb_groupself as GS
import test_groupself_bulk as F


def _norm(v):
    return F._norm(v)


def test_pep249_over_rows():
    db, con, _seg = F._fixture()
    q = "SELECT s, COUNT(*) AS c FROM t GROUP BY s ORDER BY c DESC, s LIMIT 50"
    want = [tuple(map(_norm, r)) for r in (lambda r: r[0] if isinstance(r, tuple) else r)(db.run(q))]
    cur = db.execute(q)
    assert [d[0] for d in cur.description] == ['s', 'c'] and cur.rowcount == len(want) == 50
    got = [cur.fetchone()] + cur.fetchmany(9) + cur.fetchmany() + cur.fetchall()
    assert [tuple(map(_norm, r)) for r in got] == want
    assert cur.fetchone() is None and cur.fetchmany(5) == [] and cur.fetchall() == []
    assert [tuple(map(_norm, r)) for r in db.execute(q)] == want             # iteration


def _cols_cursor(q):
    """groupself's columnar answer for q, as db.execute would wrap it"""
    db, _con, seg = F._fixture()
    spec = GS.detect(seg, sqlglot.parse_one(q, read='duckdb'), {})
    assert spec is not None
    WR.WANT[0] = True
    try:
        out = GS.execute(seg, spec)
    finally:
        WR.WANT[0] = False
    assert out is not None and isinstance(out[0], WR.Columns), q
    return WR.Cursor(out[0], out[1], db=db)


def test_columns_text_time_int_and_every_fetch_form():
    _db, con, _seg = F._fixture()
    for k, lit in (('s', "'v17'"), ('tm', "'2013-07-01 00:37:00'"), ('big', '7'), ('d', "'2000-01-14'")):
        q = "SELECT %s, COUNT(*) FROM t WHERE %s <> %s GROUP BY %s ORDER BY COUNT(*) DESC" % (k, k, lit, k)
        want = [tuple(map(_norm, r)) for r in con.execute(q + ", %s" % k).fetchall()]
        cur = _cols_cursor(q)
        assert cur.rowcount == len(want) > GS._NOLIM_MAX
        part = cur.fetchmany(1000) + cur.fetchmany(3) + [cur.fetchone()]
        rest = cur.fetchall()
        assert [tuple(map(_norm, r)) for r in part + rest] == want, k
        cur = _cols_cursor(q)                                       # columnar forms
        cur.fetchmany(10)
        npd = cur.fetchnumpy()
        names = list(npd)
        assert len(names) == 2 and len(npd[names[0]]) == len(want) - 10
        assert [_norm(v) for v in npd[names[1]].tolist()] == [r[1] for r in want[10:]]
        tbl = _cols_cursor(q).arrow()
        assert tbl.num_rows == len(want) and tbl.column(1).to_pylist() == [r[1] for r in want]
        rdr = _cols_cursor(q).fetch_record_batch(rows_per_batch=4096)
        sizes = [b.num_rows for b in rdr]
        assert sum(sizes) == len(want) and max(sizes) == 4096
        df = _cols_cursor(q).df()
        assert len(df) == len(want) and list(df.iloc[:, 1]) == [r[1] for r in want]


def test_execute_answers_equal_run_and_want_never_leaks():
    db, _con, _seg = F._fixture()
    for q in ("SELECT s, COUNT(*) FROM t WHERE s <> 'v17' GROUP BY s ORDER BY COUNT(*) DESC",
              "SELECT COUNT(*) FROM t WHERE big > 5",
              "SELECT big FROM t WHERE big IN (SELECT big FROM t WHERE id < 10)"):
        r = db.run(q)
        want = sorted(tuple(map(_norm, x)) for x in (r[0] if isinstance(r, tuple) else r))
        assert sorted(tuple(map(_norm, x)) for x in db.execute(q).fetchall()) == want, q
        assert WR.WANT[0] is False
        r2 = db.run(q)
        assert isinstance(r2[0] if isinstance(r2, tuple) else r2, list), q          # run never answers in columns


def test_echo_answers_as_a_constant_column():
    import test_echo_disk as EF
    import controller
    db, con = EF._fixture()
    for q in ("SELECT a FROM t WHERE a = 31", "SELECT s FROM t WHERE s = 'abc' LIMIT 1000"):
        n = len(con.execute(q).fetchall())
        cur = db.execute(q)
        assert controller._SERVED[0] in ('echo', 'dict_count'), (q, controller._SERVED[0])
        assert cur.rowcount == n
        first = cur.fetchmany(10)
        rest = cur.fetchall()
        e = con.execute(q).fetchall()
        assert [tuple(map(_norm, r)) for r in first + rest] == [tuple(map(_norm, r)) for r in e], q
        npd = db.execute(q).fetchnumpy()
        assert len(list(npd.values())[0]) == n
