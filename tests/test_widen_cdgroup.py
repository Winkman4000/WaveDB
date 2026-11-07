"""cdgroup FOR ANY `<>`: COUNT(DISTINCT u) GROUP BY key WHERE key <> lit ORDER BY that count DESC LIMIT k, for any
literal of the key's kind -- the old '' on a sparse key whose default is '', the stored default of a sparse key, a
value that exists, a value the key never holds, keys with NULLs, keys not stored sparse, int keys. cdgroup sits
inside the general scan behind reads that usually take these shapes, so it is called directly: detect then execute,
against DuckDB (ties at the LIMIT cut legal). Forms it cannot answer from a code filter decline (None), never wrong."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import sqlglot
import test_sparse_default as SD

_FX2 = None


def _own():
    """a sparse text key p whose default is '' (85%), a sparse int key q whose default is 0, users u, users with NULLs uu"""
    global _FX2
    if _FX2 is not None:
        return _FX2
    import uuid, tempfile, duckdb, wdb_encode
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from wdb_db import Database
    n = 300_000
    rng = np.random.default_rng(13)
    p = np.array(['p%03d' % v for v in rng.integers(0, 400, n)], dtype=object)
    p[rng.random(n) < 0.85] = ''
    q = rng.integers(1, 300, n).astype(np.int64)
    q[rng.random(n) < 0.85] = 0
    u = rng.integers(0, 5000, n).astype(np.int64)
    uu = [None if x < 0.2 else int(y) for x, y in zip(rng.random(n).tolist(), u.tolist())]
    dd = os.path.join(tempfile.gettempdir(), 'wcd_%s' % uuid.uuid4().hex[:8]); os.makedirs(dd, exist_ok=True)
    try:
        import helpers as _H; _H.register_dir(dd)
    except Exception:
        pass
    pth = os.path.join(dd, 't.parquet')
    pq.write_table(pa.table({'p': pa.array(p.tolist(), pa.string()), 'q': q, 'u': u, 'uu': pa.array(uu, pa.int64())}), pth)
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % pth)
    db = Database.create(dd)
    db.cat.add_table('t', [['p', 'str'], ['q', 'int'], ['u', 'int'], ['uu', 'int']])
    saved = {k9: os.environ.pop(k9, None) for k9 in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pth, os.path.join(dd, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    _FX2 = (db, con)
    return _FX2


def _q(key, lit, u='u', k=5):
    return "SELECT %s, COUNT(DISTINCT %s) AS du FROM t WHERE %s <> %s GROUP BY %s ORDER BY du DESC LIMIT %d" % (
        key, u, key, lit, key, k)


def _direct(db, q):
    import wdb_cdgroup, wdb_join
    seg = wdb_join._solo_segment(db, 't')[0]
    spec = wdb_cdgroup.detect(seg, sqlglot.parse_one(q), {})
    if spec is None:
        return None
    r = wdb_cdgroup.execute(seg, spec)
    return None if r is None else SD._rows(r[0])


def _serves(fx, q):
    db, con = fx
    got = _direct(db, q)
    assert got is not None, ('cdgroup declined', q)
    assert SD._legal(q, got, con), (q, got, con.execute(q).fetchall())


def test_old_case_planes():
    """`p <> ''` on a sparse key whose default is '': still the planes (neq_rows hands them, no pass over the codes)"""
    import wdb_policies, wdb_join
    fx = _own()
    seg = wdb_join._solo_segment(fx[0], 't')[0]
    assert seg.cols['p'].get('code_enc') in (8, 9) and wdb_policies.sparse_default_is(seg, 'p', ''), seg.cols['p']
    import numpy as np
    calls = []
    real = seg._raw_codes
    seg._raw_codes = lambda nm: (calls.append(nm), real(nm))[1]
    try:
        _serves(fx, _q('p', "''"))
    finally:
        del seg._raw_codes
    assert 'p' not in calls, ('the old case scanned the key codes', calls)


def test_sparse_key_other_literals():
    fx = _own()
    for lit in ("'p007'", "'zz'"):
        _serves(fx, _q('p', lit))


def test_int_key():
    fx = _own()
    for lit in ('0', '17', '-1', '99999'):
        _serves(fx, _q('q', lit))


def test_users_with_nulls():
    """COUNT(DISTINCT uu) does not count NULL"""
    fx = _own()
    for key, lit in (('p', "''"), ('q', '0'), ('p', "'zz'")):
        _serves(fx, _q(key, lit, u='uu'))


def test_shared_fixture_forms():
    fx = SD._fixture()
    for key, lit in (('s', "''"), ('s', "'x'"), ('s', "'v7'"), ('s', "'zz'"), ('n', "''"), ('n', "'n3'"),
                     ('nn', "''"), ('nn', "'m2'"), ('nn', "'zz'"), ('g', "'g2'"), ('g', "'g0'"), ('g', "'zz'"),
                     ('c', '7'), ('c', '-5'), ('k', '1')):
        _serves(fx, _q(key, lit))


def test_declines_other_kind():
    """a literal of the other kind (a number against text, text against numbers) is not a code filter: decline"""
    db, _ = _own()
    assert _direct(db, _q('p', '3')) is None
    assert _direct(db, _q('q', "'3'")) is None
    db2, _ = SD._fixture()                      # a time key: neq_rows is no code filter for it
    assert _direct(db2, _q('d', "'2013-07-01'")) is None


def bench_old_case(reps=20):
    db, _ = _own()
    q = _q('p', "''", k=10)
    _direct(db, q)
    t0 = time.perf_counter()
    for _ in range(reps):
        _direct(db, q)
    return (time.perf_counter() - t0) / reps * 1000


if __name__ == '__main__':
    fails = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        try:
            t(); print('PASS', t.__name__)
        except Exception as e:
            fails += 1; print('FAIL', t.__name__, repr(e)[:400])
    print('old `<> \'\'` case: %.2f ms/query' % bench_old_case())
    print('%d/%d passed' % (len(tests) - fails, len(tests)))
    sys.exit(1 if fails else 0)
