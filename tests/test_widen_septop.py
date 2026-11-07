"""wdb_septop widened (2026-11-07): `SELECT kcol, tcol, COUNT(*) WHERE tcol <> lit GROUP BY kcol, tcol ORDER BY c DESC
LIMIT k` where tcol is stored sparse and lit is its stored default, whatever that default is (the planes, no scan).
Every other literal -- a value that exists, a value the column never holds, a column with NULLs, a column with no '',
an int column -- septop declines and db.run answers with another read. Every answer DuckDB's (ties at the LIMIT cut
legal); the old case (`<> ''` on a sparse column whose default is '') must still read only the planes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

PLANES = [('s', "'x'")]           # s is stored sparse, default 'x'
FORMS = PLANES + [('s', "''"), ('s', "'v7'"), ('s', "'zz'"), ('g', "'g2'"), ('g', "'zz'"), ('nn', "''"),
                  ('nn', "'m3'"), ('nn', "'zz'"), ('n', "''"), ('n', "'n5'"), ('c', '62'), ('c', '-1'), ('u', '5')]
_FX2 = None
N2 = 400000


def _q(col, lit, key='k', lim=5):
    return ("SELECT %s, %s, COUNT(*) AS cc FROM %s WHERE %s <> %s GROUP BY %s, %s ORDER BY cc DESC LIMIT %d"
            % (key, col, 't', col, lit, key, col, lim))


def _direct(seg, q):
    import sqlglot, wdb_septop
    spec = wdb_septop.detect(seg, sqlglot.parse_one(q), None)
    if spec is None:
        return None
    r = wdb_septop.execute(seg, spec)
    return None if r is None else r[0]


def _fixture2():
    """its own table t (_legal reads `FROM t`), N2 rows: k (3 values), p stored sparse with default '' (80% '',
    'p0'..'p999' scattered, p0..p19 heavier) -- the old case"""
    global _FX2
    if _FX2 is not None:
        return _FX2
    import uuid, tempfile, duckdb, wdb_encode
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from wdb_db import Database
    rng = np.random.default_rng(11)
    i = np.arange(N2, dtype=np.int64)
    h = rng.random(N2)
    hv = np.where(rng.random(N2) < 0.3, rng.integers(0, 20, N2), rng.integers(0, 1000, N2))
    p = np.where(h < 0.80, '', np.char.add('p', hv.astype(str)))
    k = ((i * 7919) % 3).astype(np.int32)
    dd = os.path.join(tempfile.gettempdir(), 'wsp_%s' % uuid.uuid4().hex[:8]); os.makedirs(dd, exist_ok=True)
    try:
        import helpers as _H; _H.register_dir(dd)
    except Exception:
        pass
    pth = os.path.join(dd, 't.parquet')
    pq.write_table(pa.table({'id': i, 'k': k, 'p': pa.array(p.tolist(), pa.string())}), pth)
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % pth)
    db = Database.create(dd)
    db.cat.add_table('t', [['id', 'int'], ['k', 'int'], ['p', 'str']])
    wdb_encode.encode(pth, os.path.join(dd, 't_0.wdb'))
    db.cat.add_segment('t', 't_0.wdb')
    _FX2 = (db, con)
    return _FX2


def test_every_form_is_duckdbs_direct():
    """the planes forms: served, DuckDB's answer; every other literal: declined"""
    import test_sparse_default as T, wdb_join, wdb_policies as P
    db, con = T._fixture()
    seg, _x = wdb_join._solo_segment(db, 't')
    assert seg.cols['s'].get('code_enc') in (8, 9) and P.sparse_default_is(seg, 's', 'x'), "s: not sparse, 'x'"
    bad = []
    for col, lit in FORMS:
        q = _q(col, lit)
        got = _direct(seg, q)
        if (col, lit) not in PLANES:
            if got is not None:
                bad.append(('served a full-pass form', q))
            continue
        if got is None:
            bad.append(('declined', q)); continue
        got = T._rows(got)
        if not T._legal(q, got, con):
            bad.append((q, got[:3], T._rows(con.execute(q).fetchall())[:3]))
    assert not bad, bad


def test_every_form_through_db_run():
    """db.run: every form is DuckDB's answer, whichever read serves it (septop sits after gridwalk, which takes the
    planes forms it can; septop's own answer is the direct test's)"""
    import test_sparse_default as T, controller
    db, con = T._fixture()
    bad = []
    for col, lit in FORMS:
        q = _q(col, lit)
        got = T._rows(db.run(q))
        if not T._legal(q, got, con):
            bad.append((controller._SERVED[0], q, got[:3]))
    assert not bad, bad


def test_declines_never_wrong():
    """a literal of the other kind, a time column, OFFSET, a high-V engine: decline (None), never a wrong answer"""
    import test_sparse_default as T, wdb_join
    db, con = T._fixture()
    seg, _x = wdb_join._solo_segment(db, 't')
    for q in (_q('s', '3'), _q('c', "'62'"), _q('tm', "'2013-07-01 00:00:07'"),
              _q('s', "''") + ' OFFSET 2', _q('s', "''", key='u')):
        assert _direct(seg, q) is None, q


def test_old_case_reads_the_planes():
    """`<> ''` on a sparse column whose default is '': the planes, never a full pass over the codes"""
    import test_sparse_default as T, wdb_join, wdb_policies as P
    db, con = _fixture2()
    seg, _x = wdb_join._solo_segment(db, 't')
    assert seg.cols['p'].get('code_enc') in (8, 9) and P.sparse_default_is(seg, 'p', ''), "p: not sparse, ''"
    raw = seg._raw_codes
    def _no_scan(col, *a, **kw):
        assert col != 'p', 'the old case scanned the codes of p'
        return raw(col, *a, **kw)
    seg._raw_codes = _no_scan
    try:
        for lim in (1, 5, 20):
            q = _q('p', "''", lim=lim)
            got = _direct(seg, q)
            assert got is not None, q
            assert T._legal(q, T._rows(got), con), (q, got[:3])
    finally:
        del seg._raw_codes
    for lit in ("'p0'", "'zz'"):                     # the other literals on the same column: another read's
        assert _direct(seg, _q('p', lit)) is None, lit
        q = _q('p', lit)
        assert T._legal(q, T._rows(db.run(q)), con), q


if __name__ == '__main__':
    for nm, fn in list(globals().items()):
        if nm.startswith('test_'):
            fn(); print('PASS', nm)
