"""THE LITERAL IN A BIG DICTIONARY (2026-10-03): on a chunked front-coded text dictionary, a literal's code is found
from the chunks' first values (read from each chunk's first bytes) and then one chunk -- equal to the whole decoded
dictionary for present values, every chunk head and its neighbours, and absent values on both sides; = and <> take
that road instead of decoding the dictionary; and the inflated chunks and heads outlive the query (tier 1, through
the shelf), unless WDB_HOT_KEEP=0. Through SQL (=, <>, IN, NOT IN, with NULLs) the answers equal DuckDB's."""
import sys, os, uuid, tempfile, shutil, subprocess, bisect, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()
WDB = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
FLOOR = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')


def _frame():
    rng = np.random.default_rng(97)
    n, V = 150_000, 60_000
    vals = np.array(['%s value %06d' % (['Drama', 'Horror', 'Sweden', 'Été', 'Ω'][i % 5], i) for i in range(V)], dtype=object)
    info = vals[rng.integers(0, V, n)]
    info[rng.random(n) < 0.05] = None
    return pd.DataFrame({'id': np.arange(n, dtype=np.int64), 'info': info})


def test_literal_lookup_by_chunk_heads():
    import duckdb, wdb_engine, wdb_wherescan as W, wdb_qmem
    from wdb_db import Database
    d = os.path.join(TMP, 'dl_' + uuid.uuid4().hex[:8]); os.makedirs(d)
    old = {k: os.environ.get(k) for k in list(FLOOR) + ['WDB_HOT_KEEP']}
    try:
        df = _frame(); pq = os.path.join(d, 't.parquet'); df.to_parquet(pq, index=False)
        db_dir = os.path.join(d, 'db')
        subprocess.run([sys.executable, WDB, 'load', db_dir, 't', pq], check=True, capture_output=True, env=dict(os.environ, **FLOOR))
        path = glob.glob(db_dir + '/t_*.wdb')[0]
        seg = wdb_engine.Segment(path); c = seg.cols['info']
        assert c['mode'] == 1 and c.get('chunked') and int(c['nch']) > 1, (c['mode'], c.get('chunked'), c.get('nch'))
        ref = seg._decode_fc(c); V = len(ref); CH = int(c['CHUNK'])
        rng = np.random.default_rng(3)
        probe = [ref[int(i)] for i in rng.integers(0, V, 200)] + [b'', b'\x00', b'\xff\xff', ref[-1] + b'!']
        for j in range((V + CH - 1) // CH):
            h = ref[j * CH]; probe += [h, h + b'\x00', h[:-1]] + ([ref[j * CH - 1]] if j else [])
        seg2 = wdb_engine.Segment(path)
        for v in probe:
            k = bisect.bisect_left(ref, v)
            assert W._code_of(seg2, 'info', v) == (k if k < V and ref[k] == v else None), v
        assert len(seg2.cols['info']['_fchead']) == (V + CH - 1) // CH        # every head read, from the first bytes
        con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % pq)
        h1 = ref[CH].decode(); mid = ref[CH + 7].decode()
        sqls = ["SELECT COUNT(*) FROM t WHERE info = '%s'" % h1,
                "SELECT COUNT(*) FROM t WHERE info = '%s'" % mid,
                "SELECT COUNT(*) FROM t WHERE info = 'absent value'",
                "SELECT COUNT(*) FROM t WHERE info <> '%s'" % mid,
                "SELECT COUNT(*) FROM t WHERE info IN ('%s', '%s', 'nope', '%s')" % (h1, mid, ref[-1].decode()),
                "SELECT COUNT(*) FROM t WHERE info NOT IN ('%s', '%s')" % (h1, mid),
                "SELECT MIN(id), MAX(id) FROM t WHERE info = '%s'" % ref[0].decode(),
                # MIN/MAX of a sequence column under a filter answered the AVERAGE (pre-existing, wdb_wherescan)
                "SELECT MIN(id), MAX(id), SUM(id), COUNT(*) FROM t WHERE info = '%s'" % mid,
                "SELECT info, MIN(id), MAX(id), COUNT(*) FROM t WHERE info IN ('%s', '%s') GROUP BY info ORDER BY info" % (h1, mid)]
        os.environ.update(FLOOR)
        dbq = Database.open(db_dir)                                    # = never decodes the dictionary (the single-
        for sql in sqls[:3] + sqls[6:8]:                               # table fused IN still does: wdb_join, later)
            dbq.run(sql)
        sq = next(ent[1] for ent in dbq._seg_cache.values() if 'info' in ent[1].cols)
        assert 'info' not in sq._tdict
        for keep in ('1', '0'):
            os.environ['WDB_HOT_KEEP'] = keep
            db = Database.open(db_dir)
            for rnd in range(2):                                       # cold, then after the end-of-query flush
                for sql in sqls:
                    r = db.run(sql); got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
                    assert got == [tuple(x) for x in con.execute(sql).fetchall()], (keep, rnd, sql, got)
            s3 = next(ent[1] for ent in db._seg_cache.values() if 'info' in ent[1].cols)
            wdb_qmem.flush(db)
            kept = s3.cols['info']['chunks']; heads = s3.cols['info'].get('_fchead') or {}
            if keep == '1':
                assert kept and heads, (len(kept), len(heads))         # tier 1: the dictionary's own pieces stay
            else:
                assert not kept and not heads, (len(kept), len(heads))
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True)
