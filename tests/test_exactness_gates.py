"""EXACTNESS GATES -- the two escapes of 2026-08-31, now permanent walls.
Gap #1: declared-pair col-vs-col (the pair-bit serve's mask drop passed the
suite). Gap #2: scalar single-table with plane serves (the hoisted partition
broke Q6's answer and 1692 tests shrugged). Both shapes vs duckdb, exact."""
import sys, os, tempfile, uuid, datetime
import numpy as np
import duckdb
import wdb_encode
from wdb_db import Database
import test_join_fast as TJF

D = lambda s: (datetime.date.fromisoformat(s) - datetime.date(1970, 1, 1)).days
DL = lambda s: "DATE '%s'" % s     # duck literal (fixture dates are DATE-typed)


def _duckq(q):
    """Int-day literals -> DATE literals for the duck side."""
    import re as _re
    return _re.sub(r'(l_\w*date\s*(?:<=|>=|<|>|=)\s*)(\d{4,5})',
                   lambda m: m.group(1) + "DATE '" + (datetime.date(1970, 1, 1)
                       + datetime.timedelta(days=int(m.group(2)))).isoformat() + "'", q)

_PDB = None
_PCON = None


def _pair_fixture():
    """lineitem re-encoded with the DECLARED CLOCK (commit,receipt)."""
    global _PDB, _PCON
    if _PDB is not None:
        return _PDB, _PCON
    db0, con = TJF._fixture()
    d0 = os.path.dirname(db0.cat.segment_paths('lineitem')[0])
    d = os.path.join(tempfile.gettempdir(), f'pairdb_{uuid.uuid4().hex[:8]}')
    _PDB = Database.create(d)
    import helpers as _H; _H.register_dir(d)
    desc = con.execute("DESCRIBE lineitem").fetchall()
    _PDB.cat.add_table('lineitem', [[c[0], TJF._wt(c[1])] for c in desc])
    seg = 'lineitem_0.wdb'
    wdb_encode.encode(os.path.join(d0, 'lineitem.parquet'), os.path.join(d, seg),
                      date_pairs=[('l_commitdate', 'l_receiptdate')])
    _PDB.cat.add_segment('lineitem', seg)
    _PCON = con
    return _PDB, _PCON


def _rows(x):
    return x[0] if isinstance(x, tuple) else x


def test_exact_pair_col_vs_col_count():
    db, con = _pair_fixture()
    for q in (
        "SELECT COUNT(*) FROM lineitem WHERE l_commitdate < l_receiptdate",
        "SELECT COUNT(*) FROM lineitem WHERE l_commitdate < l_receiptdate "
        "AND l_receiptdate >= %d AND l_receiptdate < %d" % (D('1994-01-01'), D('1995-01-01')),
        "SELECT COUNT(*) FROM lineitem WHERE l_receiptdate >= l_commitdate "
        "AND l_shipdate < l_commitdate AND l_receiptdate < %d" % D('1996-06-01'),
    ):
        w = _rows(db.run(q)); e = con.execute(_duckq(q)).fetchall()
        assert int(w[0][0]) == int(e[0][0]), f"{q}\n wave={w[0][0]} duck={e[0][0]}"


def test_exact_pair_group_survivors():
    db, con = _pair_fixture()
    q = ("SELECT l_shipmode, COUNT(*) FROM lineitem WHERE l_shipmode IN ('MAIL','SHIP') "
         "AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate "
         "AND l_receiptdate >= %d AND l_receiptdate < %d "
         "GROUP BY l_shipmode ORDER BY l_shipmode" % (D('1994-01-01'), D('1995-01-01')))
    w = _rows(db.run(q)); e = con.execute(_duckq(q)).fetchall()
    got = [(str(a[0]), int(a[1])) for a in w]
    exp = [(str(b[0]), int(b[1])) for b in e]
    assert got == exp, f"{q}\n wave={got}\n duck={exp}"


def test_exact_scalar_plane_serves_q6_shape():
    db, con = TJF._fixture()
    for q in (
        "SELECT SUM(l_extendedprice * l_discount) FROM lineitem "
        "WHERE l_shipdate >= %d AND l_shipdate < %d "
        "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24" % (D('1994-01-01'), D('1995-01-01')),
        "SELECT COUNT(*) FROM lineitem WHERE l_shipdate >= %d AND l_shipdate < %d "
        "AND l_quantity < 24" % (D('1994-01-01'), D('1995-01-01')),
        "SELECT SUM(l_quantity) FROM lineitem WHERE l_shipdate <= %d" % D('1998-09-02'),
    ):
        w = _rows(db.run(q)); e = con.execute(_duckq(q)).fetchall()
        wv, ev = float(w[0][0]), float(e[0][0])
        assert abs(wv - ev) <= max(1e-6 * abs(ev), 1e-6), f"{q}\n wave={wv} duck={ev}"


def test_exact_grouped_plane_serves_q1_shape():
    db, con = TJF._fixture()
    q = ("SELECT l_returnflag, l_linestatus, SUM(l_quantity), COUNT(*) FROM lineitem "
         "WHERE l_shipdate <= %d GROUP BY l_returnflag, l_linestatus "
         "ORDER BY l_returnflag, l_linestatus" % D('1998-09-02'))
    w = _rows(db.run(q)); e = con.execute(_duckq(q)).fetchall()
    got = [(str(a[0]), str(a[1]), float(a[2]), int(a[3])) for a in w]
    exp = [(str(b[0]), str(b[1]), float(b[2]), int(b[3])) for b in e]
    assert len(got) == len(exp), f"nG={len(got)} nE={len(exp)}"
    for g, x in zip(got, exp):
        assert g[0] == x[0] and g[1] == x[1] and abs(g[2] - x[2]) < 1e-6 and g[3] == x[3], f"{g} vs {x}"
