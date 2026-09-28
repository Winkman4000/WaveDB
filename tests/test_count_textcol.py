"""COUNT(column) over a text (or float, or date) column through the dictionary-arithmetic scalar
path (wdb_join._exact_scalar): the non-null count is the codes below the dictionary's end (a NULL is
code V - 1). It used to build the integer value table for every COUNT and raised on a text dictionary
(found 2026-09-28 by the shelves' Database tests). Oracle: DuckDB."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb
from wdb_db import Database


def _db():
    return Database.create(os.path.join(tempfile.gettempdir(), f'cnt_{uuid.uuid4().hex[:8]}'))


def test_count_text_float_date_columns_with_and_without_nulls():
    db = _db()
    try:
        db.run("CREATE TABLE t (s VARCHAR, f DOUBLE, d DATE, n VARCHAR, x INT)")
        rows = []
        for i in range(400):
            s = None if i % 7 == 0 else "s%d" % (i % 37)
            f = None if i % 11 == 0 else (i % 13) * 0.5
            d = None if i % 5 == 0 else "2013-07-%02d" % (1 + i % 28)
            rows.append((s, f, d, "v%d" % (i % 9), i))
        lit = lambda v: "NULL" if v is None else ("'%s'" % v if isinstance(v, str) else repr(v))
        db.run("INSERT INTO t VALUES " + ",".join("(%s, %s, %s, %s, %d)" % (lit(s), lit(f),
               lit(d), lit(n), x) for s, f, d, n, x in rows))
        con = duckdb.connect()
        con.execute("CREATE TABLE t (s VARCHAR, f DOUBLE, d DATE, n VARCHAR, x INT)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?)", rows)
        for sql in ["SELECT COUNT(s), COUNT(*) FROM t", "SELECT COUNT(n) FROM t", "SELECT COUNT(f), COUNT(d) FROM t",
                    "SELECT COUNT(s), COUNT(n), COUNT(f), COUNT(d), COUNT(x), COUNT(*) FROM t"]:
            got = [tuple(int(v) for v in r) for r in db.run(sql)[0]]
            assert got == [tuple(int(v) for v in r) for r in con.execute(sql).fetchall()], sql
    finally:
        shutil.rmtree(db.cat.dbdir, ignore_errors=True)
