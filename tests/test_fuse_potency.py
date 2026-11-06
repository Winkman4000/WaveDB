"""The fused engine's conjunct order is MEASURED (2026-10-21): a text conjunct's keep share comes from the potency
sample through its code LUT (the census until 2026-10-31; it was a flat 0.5), and a conjunct over two columns, or one the census cannot judge, from the
rows the pred will see (wdb_join._potency9; the witness is wdb_join._POT9_LAST). Answers never depend on it."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import wdb_join
import test_join_fast as J

Q_TEXT = ("SELECT l_returnflag, COUNT(*) FROM lineitem WHERE l_shipmode = 'AIR' AND l_quantity > 5 "
          "AND l_comment LIKE '%furious%' GROUP BY l_returnflag ORDER BY l_returnflag")
Q_TWO = ("SELECT l_returnflag, COUNT(*) FROM lineitem, orders WHERE l_orderkey = o_orderkey "
         "AND l_shipdate < l_commitdate AND o_orderstatus = 'F' GROUP BY l_returnflag ORDER BY l_returnflag")


def _wit(q):
    db, con = J._fixture()
    wdb_join._POT9_LAST[0] = []
    got = [tuple(r) for r in db.run(q)[0]]
    exp = [tuple(r) for r in con.execute(q).fetchall()]
    assert got == exp, (q, got[:3], exp[:3])
    return {w[0]: w for w in wdb_join._POT9_LAST[0]}


def _share(where):
    _db, con = J._fixture()
    n = con.execute("SELECT COUNT(*) FROM lineitem").fetchone()[0]
    return con.execute("SELECT COUNT(*) FROM lineitem, orders WHERE l_orderkey = o_orderkey AND " + where).fetchone()[0] / n


def test_text_conjunct_scored_from_the_keep_sample():
    """THE SAMPLED KEEP SHARE (2026-10-31): with the sidecar switch off the code LUT's share comes from the potency
    sample -- every row of this small fixture, so exact; with the switch on (the suite's default) and with the
    sample switched off, from the census"""
    for env, oks, how in (('0', True, 'sample-keep'), ('1', True, 'census'), ('0', False, 'census')):
        old = (os.environ.get('WDB_SIDECARS'), wdb_join._KEEP_SAMPLE[0])
        try:
            os.environ['WDB_SIDECARS'] = env; wdb_join._KEEP_SAMPLE[0] = oks
            _db9 = J._fixture()[0]                                      # not yet counted (an earlier test may have)
            _db9.open_segment(_db9.cat.segment_paths('lineitem')[0], 'lineitem').__dict__.pop('_ccounts', None)
            sm = _wit(Q_TEXT)["l_shipmode = 'AIR'"]
            assert sm[3] == how, (env, oks, sm)
            assert abs(sm[1] - (1.0 - _share("l_shipmode = 'AIR'"))) < 1e-4, sm    # exact (the witness rounds to 4 places), not 0.5
        finally:
            if old[0] is None: os.environ.pop('WDB_SIDECARS', None)
            else: os.environ['WDB_SIDECARS'] = old[0]
            wdb_join._KEEP_SAMPLE[0] = old[1]


def test_unjudged_conjuncts_sampled_on_the_rows_the_pred_sees():
    old = wdb_join._POT9_MIN_ROWS
    wdb_join._POT9_MIN_ROWS = 0                     # the fixture is small: measure anyway
    try:
        w = _wit(Q_TWO)
        two = w['l_shipdate < l_commitdate']
        assert two[3] == 'sample', two
        # under 64K rows the sample is every row: the share is exact
        assert abs(two[1] - (1.0 - _share('l_shipdate < l_commitdate'))) < 1e-4, two
        lk = _wit(Q_TEXT)["l_comment LIKE '%furious%'"]
        assert lk[3] == 'sample' and abs(lk[1] - (1.0 - _share("l_comment LIKE '%furious%'"))) < 1e-4, lk
    finally:
        wdb_join._POT9_MIN_ROWS = old


def test_same_answers_with_the_measurement_off():
    old = wdb_join._FUSE_POT[0]
    try:
        wdb_join._FUSE_POT[0] = False
        w = _wit(Q_TEXT)
        assert w["l_shipmode = 'AIR'"][1] == 0.5                  # the old flat guess for text
        _wit(Q_TWO)
    finally:
        wdb_join._FUSE_POT[0] = old
