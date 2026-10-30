"""THE STORED START (2026-10-30): the potency pick lets a filter whose passing rows are already stored -- '<>' the
default of an enc-8 column, as SearchPhrase <> '' -- start whenever it keeps no more rows than the ladder's driver,
not only under half its rows. s LIKE '%yz' keeps ~17% and p <> '' ~10%: p starts. Two LIKEs as close stay on the
ladder's driver; '<>' a non-default value is not a stored start. Answers equal DuckDB's and the pick-off answers."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np

_FX = None


def _fixture():
    """200K rows: p blank on 90% of rows (stored as the sparse dress, enc 8), s twelve values (two end in 'yz'),
    b 'v<n>' + 'abc' on a fifth of rows, g a 40-value key"""
    global _FX
    if _FX is not None:
        return _FX
    import duckdb, wdb_encode
    from wdb_db import Database
    con = duckdb.connect()
    con.execute("""CREATE TABLE t AS SELECT i AS id,
        ['abc', 'xabcz', 'qyz', 'ab', 'mm', 'jj', 'a1c', 'zzz', 'x', 'yy5', 'abyz', 'k'][1 + ((i * 7919) % 12)] AS s,
        'v' || ((i * 104729) % 30011)::VARCHAR || CASE WHEN i % 5 = 0 THEN 'abc' ELSE 'xy' END AS b,
        'g' || ((i * 37) % 40)::VARCHAR AS g,
        CASE WHEN (i * 7) % 10 = 0 THEN 'p' || ((i * 13) % 500)::VARCHAR ELSE '' END AS p
        FROM range(200000) r(i)""")
    d = os.path.join(tempfile.gettempdir(), 'sst_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db = Database.create(d)
    db.cat.add_table('t', [['id', 'int'], ['s', 'str'], ['b', 'str'], ['g', 'str'], ['p', 'str']])
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    os.environ['WDB_E8_FORCE'] = '1'
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        os.environ.pop('WDB_E8_FORCE', None)
        for k9, v9 in saved.items():
            if v9 is not None:
                os.environ[k9] = v9
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con, db.open_segment(db.cat.segment_paths('t')[0], 't'))
    return _FX


def _rows(res):
    return [tuple(v.decode() if isinstance(v, bytes) else v for v in r) for r in res[0]]


def _legal(sql, w, con):
    """a tied LIMIT is legal: the counts are DuckDB's top-k in order, every row is DuckDB's row for its key"""
    d = con.execute(sql).fetchall()
    assert [int(x[2]) for x in w] == [int(x[2]) for x in d], (sql, w[:3], d[:3])
    inner = sql[:sql.upper().index(' ORDER BY')]
    truth = {x[0]: tuple(x) for x in con.execute(inner).fetchall()}
    for x in w:
        assert tuple(x) == truth.get(x[0]), (sql, x, truth.get(x[0]))


def test_stored_start():
    db, con, seg = _fixture()
    import sqlglot, controller, wdb_wherescan as WS
    assert seg.cols['p'].get('code_enc') == 8, seg.cols['p'].get('code_enc')
    q1 = "SELECT g, MIN(s), COUNT(*) AS c FROM t WHERE s LIKE '%yz' AND p <> '' GROUP BY g ORDER BY c DESC LIMIT 10"
    q2 = "SELECT g, MIN(s), COUNT(*) AS c FROM t WHERE b LIKE '%abc' AND s LIKE '%yz' GROUP BY g ORDER BY c DESC LIMIT 10"
    sp1 = WS.detect(seg, sqlglot.parse_one(q1), None)
    ip = [i for i, e in enumerate(sp1['eqs']) if e[0] == 'p'][0]
    assert WS._stored_start(seg, sp1, ('eq', ip))
    sp3 = WS.detect(seg, sqlglot.parse_one(q1.replace("p <> ''", "p <> 'p7'")), None)
    ip3 = [i for i, e in enumerate(sp3['eqs']) if e[0] == 'p'][0]
    assert not WS._stored_start(seg, sp3, ('eq', ip3))
    old = WS._POT_SWITCH
    for q, want in ((q1, 'p'), (q2, 'b')):
        WS._POT_LAST[0] = None
        controller.plan_epoch_bump()
        w = _rows(db.run(q))
        assert controller._SERVED[0] == 'wherescan', (q, controller._SERVED[0])
        pk = WS._POT_LAST[0]
        assert pk is not None, q
        sp = WS.detect(seg, sqlglot.parse_one(q), None)
        if want == 'p':
            kind, i = pk[0]
            assert (kind, sp['eqs'][i][0]) == ('eq', 'p'), (q, pk)
            assert pk[2]['eq%d' % i] >= old * pk[2]['like0'], ('not in the band the stored start decides', pk)
        else:
            kind, i = pk[1]                       # the ladder's driver stays
            assert pk[0] != pk[1] and (kind, sp['likes'][i][0]) == ('like', 'b'), (q, pk)
            assert pk[2]['like1'] >= old * pk[2]['like0'], ('not in the band', pk)
        _legal(q, w, con)
        WS._POTENCY[0] = False
        try:
            controller.plan_epoch_bump()
            assert _rows(db.run(q)) == w, q
        finally:
            WS._POTENCY[0] = True
