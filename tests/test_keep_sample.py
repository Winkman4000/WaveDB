"""THE SAMPLED KEEP SHARE (2026-10-31): wdb_join._keep_share reads a keep table's share from wherescan's potency
sample instead of a census of every row. On 400K rows (the sample: 32 blocks of 2,048) it agrees with the census
within a few points for dense, sparse and range keeps; on a segment smaller than the sample it is the census
exactly; switched off, or on a non-dictionary column, it declines (None) so the caller counts. Join answers equal
DuckDB's with it on and off."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, contextlib

_FX = None


@contextlib.contextmanager
def _off():
    """the sidecar switch off (the suite runs with it on, where the persisted census is used instead)"""
    old = os.environ.get('WDB_SIDECARS')
    os.environ['WDB_SIDECARS'] = '0'
    try:
        yield
    finally:
        if old is None: os.environ.pop('WDB_SIDECARS', None)
        else: os.environ['WDB_SIDECARS'] = old


def _fixture():
    global _FX
    if _FX is not None:
        return _FX
    import duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        CAST(hash(i // 16, 1) % 2400 AS INTEGER) AS d,
        ['AIR', 'MAIL', 'SHIP', 'RAIL', 'TRUCK', 'FOB', 'REG AIR'][1 + CAST(hash(i, 2) % 7 AS BIGINT)] AS m,
        CAST(hash(i, 3) % 50 AS INTEGER) + 1 AS q
        FROM range(400000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'ksm_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['d', 'int'], ['m', 'str'], ['q', 'int']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con, db.open_segment(db.cat.segment_paths('t')[0], 't'))
    return _FX


def _census_share(seg, col, keep):
    cnt = np.asarray(seg.code_counts(col))
    return float(cnt[np.asarray(keep, bool)[:cnt.size]].sum()) / cnt.sum()


def test_sample_share_against_the_census():
    with _off():
        _sample_share_against_the_census()


def _sample_share_against_the_census():
    import wdb_join
    _db, _con, seg = _fixture()
    rng = np.random.default_rng(1031)
    checked = 0
    for col in ('d', 'm', 'q'):
        c = seg.cols[col]
        if c.get('mode') not in (0, 1, 2):
            continue
        V = int(c['V'])
        td = np.asarray(seg._typed_dict(col)) if col != 'm' else None
        keeps = [rng.random(V + 1) < 0.5, rng.random(V + 1) < 0.05]
        if td is not None and td.dtype.kind in 'iu':
            k = np.zeros(V + 1, bool); k[:len(td)] = td < np.quantile(td, 0.3); keeps.append(k)
        for keep in keeps:
            seg.__dict__.pop('_ccounts', None)                     # not yet counted: the sample answers
            sh = wdb_join._keep_share(seg, col, keep)
            assert sh is not None, col
            cs = _census_share(seg, col, keep)
            assert abs(sh - cs) < 0.03, (col, sh, cs)
            checked += 1
    assert checked >= 6, checked


def test_small_segment_is_the_census_and_the_switch_declines():
    with _off():
        _small_segment()
    import wdb_join, test_join_fast as J
    db, _con = J._fixture()
    seg = db.open_segment(db.cat.segment_paths('lineitem')[0], 'lineitem')
    keep = np.ones(int(seg.cols['l_shipmode']['V']) + 1, bool)
    if __import__('wdb_sidecar').births_on(os.path.dirname(seg.path)):
        assert wdb_join._keep_share(seg, 'l_shipmode', keep) is None   # the switch on: the persisted census answers


def _small_segment():
    import wdb_join, test_join_fast as J
    db, _con = J._fixture()
    seg = db.open_segment(db.cat.segment_paths('lineitem')[0], 'lineitem')
    assert int(seg.N) <= 32 * 2048, seg.N                      # the sample is every row
    for col in ('l_shipmode', 'l_returnflag'):
        V = int(seg.cols[col]['V'])
        keep = np.zeros(V + 1, bool); keep[0] = True
        seg.__dict__.pop('_ccounts', None)
        assert abs(wdb_join._keep_share(seg, col, keep) - _census_share(seg, col, keep)) < 1e-12, col
    old = wdb_join._KEEP_SAMPLE[0]
    try:
        wdb_join._KEEP_SAMPLE[0] = False
        assert wdb_join._keep_share(seg, 'l_shipmode', keep) is None
    finally:
        wdb_join._KEEP_SAMPLE[0] = old


def test_join_answers_both_ways():
    import wdb_join, controller, test_join_fast as J
    db, con = J._fixture()
    qs = ["SELECT l_returnflag, COUNT(*) FROM lineitem, orders WHERE l_orderkey = o_orderkey AND l_shipmode = 'AIR' "
          "AND l_quantity > 5 AND o_orderstatus = 'F' GROUP BY l_returnflag ORDER BY l_returnflag",
          "SELECT l_shipmode, SUM(l_extendedprice) FROM lineitem, orders WHERE l_orderkey = o_orderkey "
          "AND l_quantity >= 10 AND l_quantity <= 30 AND o_orderpriority = '1-URGENT' GROUP BY l_shipmode ORDER BY l_shipmode"]
    old = wdb_join._KEEP_SAMPLE[0]
    try:
      with _off():
        for q in qs:
            exp = [tuple(r) for r in con.execute(q).fetchall()]
            for on in (True, False):
                wdb_join._KEEP_SAMPLE[0] = on
                controller.plan_epoch_bump()
                got = [tuple(r) for r in db.run(q)[0]]
                assert len(got) == len(exp), (q, on, got[:3], exp[:3])
                for a, b in zip(got, exp):
                    for x, y in zip(a, b):
                        num = not isinstance(y, (str, bytes)) and y is not None
                        assert (abs(float(x) - float(y)) <= 1e-6 * max(1.0, abs(float(y)))) if num else x == y, (q, on, a, b)
    finally:
        wdb_join._KEEP_SAMPLE[0] = old
