"""THE PART KEY (2026-11-07): diskpair's derived time key was EXTRACT(minute) only, read off the dictionary as seconds.
Now any EXTRACT part a dictionary value alone decides -- second, minute, hour, day, month, quarter, year, dow, isodow,
doy, and their function forms -- over a stair datetime column, in the column's own time unit, alone or beside other
keys (two parts too). NULL time columns and the other parts (week, epoch, sub-second) decline. Every answer DuckDB's."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import sqlglot
import wdb_diskpair as DP

PARTS = ('second', 'minute', 'hour', 'day', 'month', 'quarter', 'year', 'dow', 'isodow', 'doy', 'dayofweek',
         'dayofyear', 'dayofmonth')
FNS = ('second', 'minute', 'hour', 'day', 'month', 'quarter', 'year', 'dayofweek', 'isodow', 'dayofyear',
       'dayofmonth')
_FX = None


def _wide():
    """300K rows in time order over 1965..1975 (across the epoch, so the floors matter): ts (seconds, odd steps,
    one tick per 3 rows), tu (microseconds with fractions, one tick per 5 rows), tn (a staircase with NULLs at the end),
    x (spread ints: framed, the block geometry), y (41 ints: not framed)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from wdb_db import Database
    N = 300000
    i = np.arange(N, dtype=np.int64)
    t0 = np.datetime64('1965-03-01T00:00:00', 's').astype(np.int64)
    ts = (t0 + (i // 3) * 3163 + ((i // 3) * 7) % 11).astype('datetime64[s]')
    tu = (t0 * 1000000 + (i // 5) * 5011000123).astype('datetime64[us]')
    tn = [None if j >= N - 5000 else v for j, v in enumerate(ts.tolist())]
    x = ((i * 2654435761) % 3000).astype(np.int32)
    y = ((i * 48271) % 41).astype(np.int32)
    tbl = pa.table({'id': i, 'ts': pa.array(ts, pa.timestamp('s')), 'tu': pa.array(tu, pa.timestamp('us')),
                    'tn': pa.array(tn, pa.timestamp('s')), 'x': x, 'y': y})
    dd = os.path.join(tempfile.gettempdir(), 'dpw_%s' % uuid.uuid4().hex[:8]); os.makedirs(dd, exist_ok=True)
    try:
        import helpers as _H; _H.register_dir(dd)
    except Exception:
        pass
    p = os.path.join(dd, 't.parquet'); pq.write_table(tbl, p)
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % p)
    db = Database.create(dd)
    db.cat.add_table('t', [['id', 'int'], ['ts', 'datetime'], ['tu', 'datetime'], ['tn', 'datetime'],
                           ['x', 'int'], ['y', 'int']])
    saved = {k9: os.environ.pop(k9, None) for k9 in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:                                         # the runner's sequence levers would store the times raw
        wdb_encode.encode(p, os.path.join(dd, 't_0.wdb'))
    finally:
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    _FX = (db, con, seg)
    return _FX


def _rows(r):
    return r[0] if isinstance(r, tuple) else r


def _norm(rows):
    return sorted((tuple(int(v) if hasattr(v, '__index__') else v for v in r) for r in rows),
                  key=lambda r: [(v is None, '' if v is None else v) for v in r])     # NULL sorts last


def _run(db, q, counted=False):
    """db.run(q); the counted form ahead of diskpair off unless counted (so diskpair itself answers)"""
    import controller
    old = os.environ.get('WDB_COUNTED_DISTINCT')
    if not counted:
        os.environ['WDB_COUNTED_DISTINCT'] = '0'
    try:
        r = db.run(q)
    finally:
        if old is None:
            os.environ.pop('WDB_COUNTED_DISTINCT', None)
        else:
            os.environ['WDB_COUNTED_DISTINCT'] = old
    return _rows(r), controller._SERVED[0]


def _check(db, con, q):
    """diskpair answers q (the counted form ahead of it off) and the answer is DuckDB's"""
    rows, who = _run(db, q)
    assert who == 'diskpair', (q, who)
    assert _norm(rows) == _norm(con.execute(q).fetchall()), q


def _stair_cols(seg, *cols):
    for c in cols:
        assert seg.stairs(c) is not None, 'the fixture did not store %s as a staircase' % c


def test_every_part_alone_wide():
    db, con, seg = _wide()
    _stair_cols(seg, 'ts', 'tu')
    for col in ('ts', 'tu'):
        for part in PARTS:
            _check(db, con, 'SELECT DISTINCT EXTRACT(%s FROM %s) AS v FROM t' % (part, col))
        for fn in FNS:
            _check(db, con, 'SELECT DISTINCT %s(%s) FROM t' % (fn, col))


def test_every_part_alone_sparse_fixture():
    import test_sparse_default as T
    db, con = T._fixture()[:2]
    seg = db.open_segment(db.cat.segment_paths('t')[0], 't')
    _stair_cols(seg, 'tm')
    for part in PARTS:
        _check(db, con, 'SELECT DISTINCT EXTRACT(%s FROM tm) FROM t' % part)


def test_parts_beside_keys():
    db, con, seg = _wide()
    for q in ('SELECT DISTINCT x, EXTRACT(hour FROM ts) AS h FROM t',
              'SELECT DISTINCT EXTRACT(second FROM tu) AS s, x FROM t',
              'SELECT DISTINCT x, EXTRACT(dow FROM ts), EXTRACT(month FROM tu) FROM t',
              'SELECT DISTINCT EXTRACT(hour FROM ts) AS h, EXTRACT(minute FROM ts) AS m FROM t',
              'SELECT DISTINCT x, year(ts) FROM t'):
        _check(db, con, q)
    import test_sparse_default as T
    db2, con2 = T._fixture()[:2]
    for q in ('SELECT DISTINCT u, EXTRACT(minute FROM tm) AS m FROM t',            # the old shape
              'SELECT DISTINCT n, EXTRACT(hour FROM tm) FROM t',
              'SELECT DISTINCT nn, EXTRACT(dow FROM tm) FROM t',                   # a key with NULLs
              'SELECT DISTINCT n, EXTRACT(second FROM tm), EXTRACT(minute FROM tm) FROM t'):
        _check(db2, con2, q)


def test_limit_offset_legal():
    db, con, seg = _wide()
    for q, full in (('SELECT DISTINCT x, EXTRACT(hour FROM ts) FROM t LIMIT 10',
                     'SELECT DISTINCT x, EXTRACT(hour FROM ts) FROM t'),
                    ('SELECT DISTINCT EXTRACT(doy FROM ts) FROM t LIMIT 7 OFFSET 3',
                     'SELECT DISTINCT EXTRACT(doy FROM ts) FROM t')):
        rows, who = _run(db, q)
        assert who == 'diskpair', (q, who)
        n = int(q.split('LIMIT ')[1].split()[0])
        allp = set(_norm(con.execute(full).fetchall()))
        got = _norm(rows)
        assert len(got) == n and len(set(got)) == n and all(r in allp for r in got), q


def test_declines():
    db, con, seg = _wide()
    for q in ('SELECT DISTINCT EXTRACT(week FROM ts) FROM t', 'SELECT DISTINCT EXTRACT(epoch FROM ts) FROM t',
              'SELECT DISTINCT EXTRACT(millisecond FROM tu) FROM t', 'SELECT DISTINCT EXTRACT(hour FROM tn) FROM t',
              'SELECT DISTINCT x, EXTRACT(minute FROM tn) FROM t'):
        assert DP.detect(seg, sqlglot.parse_one(q, read='duckdb'), {}) is None, q
        try:
            rows, who = _run(db, q)
        except NotImplementedError:
            continue                             # the general path has no such part either: an error, not a wrong answer
        assert who != 'diskpair', (q, who)
        if 'tn' not in q:                        # tn's parts are the general path's (it reads them as seconds)
            assert _norm(rows) == _norm(con.execute(q).fetchall()), (q, who)


if __name__ == '__main__':
    n = 0
    for nm, fn in list(globals().items()):
        if nm.startswith('test_') and callable(fn):
            fn(); n += 1; print('ok', nm)
    print('%d passed' % n)
