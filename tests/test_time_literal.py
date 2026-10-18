"""A time literal in a filter (2026-10-17, the Q14 board): WHERE EventTime <> '2013-07-21 19:37:51' reached the
dictionary lookup as text, int() failed, and the failed lookup read as "absent" -- pairagg's '<>' kept every
row (a wrong answer) and the composed reads declined. The literal is now parsed to the column's stored unit
(wdb_wherescan._time_units) in the one lookup the reads share. Checked for LEGALITY against DuckDB: the counts
are DuckDB's top-k counts and every row is a real group of the filtered rows with its true count."""
import sys, os, uuid, tempfile, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, duckdb
from wdb_db import Database
import wdb_encode, wdb_join, wdb_wherescan as WS, controller

N = 200_000
H = '2013-07-21 19:37:51'
_D = None


def _fixture():
    global _D
    if _D is not None:
        return _D
    con = duckdb.connect()
    d = os.path.join(tempfile.gettempdir(), 'tlit_%s' % uuid.uuid4().hex[:8]); os.makedirs(d, exist_ok=True)
    # ts: one heavy time (30% of rows, the board's top value) over a day of seconds; k tiny, s small, w text
    con.execute("""CREATE TABLE t AS SELECT i AS id, i %% 5 AS k, (i * 31) %% 300 AS s,
        CASE WHEN (i * 7) %% 10 < 3 THEN TIMESTAMP '%s'
             ELSE TIMESTAMP '2013-07-21 00:00:00' + INTERVAL ((i * 7907) %% 86400) SECOND END AS ts,
        CASE WHEN i %% 4 = 0 THEN 'x' WHEN i %% 4 = 1 THEN 'y' ELSE '' END AS w
        FROM range(%d) t(i)""" % (H, N))
    db = Database.create(d)
    pq = os.path.join(d, 't.parquet')
    con.execute("COPY (SELECT * FROM t ORDER BY id) TO '%s' (FORMAT parquet)" % pq)
    db.cat.add_table('t', [['id', 'int'], ['k', 'int'], ['s', 'int'], ['ts', 'datetime'], ['w', 'str']])
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    _D = (db, con, d)
    return _D


def _norm(v, unit):
    if isinstance(v, (bytes, bytearray)):
        return v.decode()
    if isinstance(v, (int, np.integer)) and unit is not None:
        return np.datetime64(int(v), unit).astype('datetime64[us]')
    if isinstance(v, (datetime.datetime, np.datetime64)):
        return np.datetime64(v).astype('datetime64[us]')
    return v.item() if hasattr(v, 'item') else v


def _legal(q, where, k1, k2, lim):
    db, con, d = _fixture()
    seg, _ = wdb_join._solo_segment(db, 't')
    rows = db.run(q); rows = rows[0] if isinstance(rows, tuple) else rows
    served = controller._SERVED[0]
    u = {c: (seg.unit(c) if c == 'ts' else None) for c in (k1, k2)}
    truth = {(_norm(x[0], None), _norm(x[1], None)): x[2] for x in con.execute(
        "SELECT %s, %s, COUNT(*) FROM t WHERE %s GROUP BY %s, %s" % (k1, k2, where, k1, k2)).fetchall()}
    want = sorted(truth.values(), reverse=True)[:lim]
    got = [int(r[2]) for r in rows]
    assert got == want, (q, served, got, want)
    for r in rows:
        key = (_norm(r[0], u[k1]), _norm(r[1], u[k2]))
        assert truth.get(key) == int(r[2]), (q, served, key, r[2])
    return served


WHERES = ("ts <> '%s'" % H, "ts = '%s'" % H, "ts <> '%s.5'" % H, "ts = '%s.5'" % H,
          "ts <> '2013-07-21'", "ts <> '2099-01-01 00:00:00'")


def test_time_literal_every_read():
    """each filter form, on a key and off the keys: the answer is legal whichever read takes it"""
    for where in WHERES:
        for k1, k2 in (('k', 'ts'), ('ts', 's'), ('w', 'ts'), ('k', 's')):
            q = 'SELECT %s, %s, COUNT(*) AS c FROM t WHERE %s GROUP BY %s, %s ORDER BY c DESC LIMIT 10' % (
                k1, k2, where, k1, k2)
            served = _legal(q, where, k1, k2, 10)
            assert served != 'general_scan', (q, served)


def test_time_units():
    """the literal in stored units: exact text, a finer literal matches nothing, numbers are native, junk raises"""
    db, con, d = _fixture()
    seg, _ = wdb_join._solo_segment(db, 't')
    unit = seg.unit('ts')
    iv = WS._time_units(seg, 'ts', H)
    assert np.datetime64(iv, unit) == np.datetime64(H.replace(' ', 'T'))
    assert WS._time_units(seg, 'ts', H + '.5') is None or unit in ('ms', 'us', 'ns')
    assert WS._time_units(seg, 'ts', 12345) == 12345
    assert WS._code_of(seg, 'ts', H) is not None
    assert WS._code_of(seg, 'ts', '2099-01-01 00:00:00') is None
    try:
        WS._time_units(seg, 'ts', 'not a time')
        assert False, 'junk literal accepted'
    except ValueError:
        pass
