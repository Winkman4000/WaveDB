"""STRING LENGTHS AS LOAD DATA (wdb_lens, Jackson 2026-09-25): the load stores every front-coded text
column's dictionary CHARACTER lengths, and -- when the operator names the column (--row-lengths) -- its
per-row character lengths in row order. Every stored length must equal the string's character count,
length aggregates must equal DuckDB's with and without the files, and a stale file must be ignored."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_lens
from wdb_db import Database

TMP = tempfile.gettempdir()
_FIX = None
SQLS = [
    "SELECT k, AVG(length(s)) AS l, COUNT(*) AS c FROM tbl WHERE s <> '' GROUP BY k HAVING COUNT(*) > 100 ORDER BY l DESC LIMIT 10",
    "SELECT k, AVG(length(s)) AS l, COUNT(*) AS c FROM tbl GROUP BY k ORDER BY l DESC LIMIT 7",
    "SELECT k, SUM(length(s)) AS l, COUNT(*) AS c FROM tbl WHERE s <> '' GROUP BY k ORDER BY l DESC LIMIT 5",
]


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(25)
    alpha = list('abcdefghijklmnopqrstuvwxyz/._-') + ['é', 'ü', '日', '本', 'ж', '😀']
    pool = [''.join(rng.choice(alpha, int(n))) for n in rng.integers(1, 120, 60000)]
    pool = sorted(set(pool))
    n = 200000
    s = np.array(pool, dtype=object)[rng.integers(0, len(pool), n)]
    s[rng.random(n) < 0.05] = ''
    df = pd.DataFrame({'k': rng.integers(0, 40, n).astype(np.int64), 's': s})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/ln_{t}.parquet'
    df.to_parquet(pq, index=False)
    d = f'{TMP}/lndb_{t}'
    db = Database.create(d)
    db.cat.add_table('tbl', [['k', 'int'], ['s', 'string']])
    keep = {k: os.environ.get(k) for k in ('WDB_ROWLEN_COLS', 'WDB_SEQ_NARROW_OK')}
    os.environ.update(WDB_ROWLEN_COLS='s', WDB_SEQ_NARROW_OK='0')
    try:
        wdb_encode.encode(pq, os.path.join(d, 'tbl_0.wdb'))
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    db.cat.add_segment('tbl', 'tbl_0.wdb')
    _FIX = (db, pq, os.path.join(d, 'tbl_0.wdb'), df)
    return _FIX


def _seg():
    from wdb_engine import Segment
    return Segment(_fixture()[2])


def test_files_written_and_exact():
    db, pq, w, df = _fixture()
    s = _seg()
    assert s.cols['s']['mode'] == 1, s.cols['s']['mode']
    assert os.path.exists(wdb_lens.dict_path(w, 's')) and os.path.exists(wdb_lens.row_path(w, 's'))
    dl = wdb_lens.dict_lens(s, 's')
    assert dl is not None
    vals = s.dict_vals('s')
    want = np.array([len(v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v) for v in vals], np.int64)
    assert np.array_equal(dl, want), 'dictionary lengths'
    rl = wdb_lens.row_lens(s, 's')
    assert np.array_equal(rl.astype(np.int64), df['s'].str.len().to_numpy(np.int64)), 'row lengths'


def test_stale_files_are_ignored():
    db, pq, w, df = _fixture()
    import struct
    from wdb_engine import Segment
    s = Segment(w)
    for p in (wdb_lens.dict_path(w, 's'), wdb_lens.row_path(w, 's')):
        raw = bytearray(open(p, 'rb').read())
        keep = bytes(raw)
        size = struct.unpack_from('<q', raw, 20)[0]
        struct.pack_into('<q', raw, 20, size + 1)          # written for a different segment
        open(p, 'wb').write(bytes(raw))
        try:
            assert wdb_lens.dict_lens(s, 's') is None if 'clen' in p else not wdb_lens.has_row_lens(s, 's')
        finally:
            open(p, 'wb').write(keep)
    assert wdb_lens.dict_lens(s, 's') is not None and wdb_lens.has_row_lens(s, 's')


def _duck(pq, sql):
    return duckdb.connect().execute(sql.replace('FROM tbl', "FROM read_parquet('%s')" % pq)).fetchall()


def _norm(rows):
    return [(int(r[0]), round(float(r[1]), 9), int(r[2])) for r in rows]


def test_aggregates_equal_duckdb_with_row_lengths():
    db, pq, w, df = _fixture()
    import wdb_lenagg
    for sql in SQLS:
        assert _norm(db.run(sql)[0]) == _norm(_duck(pq, sql)), sql
    import sqlglot
    spec = wdb_lenagg.detect(_seg(), sqlglot.parse_one(SQLS[0]), {})
    wdb_lenagg._PF.clear()                           # detect alone started the key thread
    assert spec is not None and spec['rowl'], 'the row-length road was not taken'


def test_aggregates_equal_duckdb_without_row_lengths():
    db, pq, w, df = _fixture()
    p = wdb_lens.row_path(w, 's'); moved = p + '.off'
    os.replace(p, moved)
    try:
        for sql in SQLS:
            assert _norm(Database.open(os.path.dirname(w)).run(sql)[0]) == _norm(_duck(pq, sql)), sql
    finally:
        os.replace(moved, p)
