"""THE BLOCKED INLINE COLUMN (2026-10-03): an inline text column written as blocks of rows (each block its rows'
byte lengths, then their text, each zstd alone, one start table) reads back byte for byte -- the whole stream,
point reads in a few blocks and across many, the object array -- at every block boundary and both length widths;
through SQL (MIN/MAX at the survivors' blocks, LIKE, =, IN) it answers as DuckDB does, and as the one-block layout
does (WDB_INLINE_BLOCK=0)."""
import sys, os, uuid, tempfile, shutil, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()
WDB = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
FLOOR = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')


def _frames(n, K):
    rng = np.random.default_rng(91)
    words = ['Shrek', 'Queen', 'alpha', 'Ωmega', 'été', 'x']
    s = np.array(['%s %07d %s' % (words[i % 6], int(v), 'y' * int(v % 13)) for i, v in
                  enumerate(rng.permutation(n))], dtype=object)
    s[[0, K - 1, K, K + 1, n - 1]] = ['', 'Aardvark', 'zzz', 'Shrek 2', 'Zulu end']
    s[5] = 'L' * 70_000                                     # a row past 65,535 bytes: the wide length width
    return pd.DataFrame({'id': np.arange(n, dtype=np.int64), 'grp': np.arange(n) % 97, 's': s})


def _load(df, d, name, block):
    pq = os.path.join(d, name + '.parquet'); df.to_parquet(pq, index=False)
    db_dir = os.path.join(d, name + '_db')
    subprocess.run([sys.executable, WDB, 'load', db_dir, 't', pq], check=True, capture_output=True,
                   env=dict(os.environ, WDB_INLINE_BLOCK=str(block), **FLOOR))
    return db_dir, pq


def test_blocked_inline_reads_back_and_answers_as_duck():
    import duckdb, wdb_engine, glob
    from wdb_db import Database
    K = 16384
    d = os.path.join(TMP, 'ib_' + uuid.uuid4().hex[:8]); os.makedirs(d)
    old = {k: os.environ.get(k) for k in FLOOR}
    try:
        for n in (K + 2, 3 * K + 517):
            df = _frames(n, K)
            ref = [x.encode() for x in df['s']]
            blk_dir, pq = _load(df, d, 'b%d' % n, K)
            one_dir, _ = _load(df, d, 'o%d' % n, 0)
            seg = wdb_engine.Segment(glob.glob(blk_dir + '/t_*.wdb')[0])
            c = seg.cols['s']
            assert c['mode'] == 5 and c.get('iblk') is not None, (c['mode'], c.get('iblk'))
            assert c['iblk'][0] == K and c['iblk'][1] == (n + K - 1) // K and c['iblk'][2] == 4
            seg1 = wdb_engine.Segment(glob.glob(one_dir + '/t_*.wdb')[0])
            assert seg1.cols['s']['mode'] == 5 and seg1.cols['s'].get('iblk') is None
            rng = np.random.default_rng(n)
            few = np.array([0, 3, K - 1, 5], np.int64)                 # one block: the point road
            assert seg.inline_at('s', few) == [ref[i] for i in few]
            if c['iblk'][1] >= 4:                                      # a quarter of the blocks or fewer
                assert '_istream' not in c and set(c['_iblk']) == {0}
            many = rng.integers(0, n, 200).astype(np.int64)            # every block: the whole stream
            assert seg.inline_at('s', many) == [ref[i] for i in many]
            blob, off = seg.inline_stream('s')
            assert off.size == n + 1 and bytes(blob[:off[-1]]) == b''.join(ref)
            assert list(seg._inline_values(c)) == ref
            seg2 = wdb_engine.Segment(seg.path)                        # the stream straight, no point read first
            b2, o2 = seg2.inline_stream('s')
            assert np.array_equal(o2, off) and bytes(b2[:o2[-1]]) == bytes(blob[:off[-1]])
            con = duckdb.connect()
            con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % pq)
            sqls = ["SELECT MIN(s), MAX(s) FROM t",
                    "SELECT MIN(s), MAX(s) FROM t WHERE id IN (1, 2, %d, %d)" % (K, K + 1),
                    "SELECT MIN(s), MAX(s), COUNT(*) FROM t WHERE grp = 7",
                    "SELECT COUNT(*) FROM t WHERE s LIKE '%%Shrek%%'",
                    "SELECT COUNT(*) FROM t WHERE s LIKE 'é%%' OR s LIKE 'Ω%%'",
                    "SELECT id FROM t WHERE s = 'Shrek 2'",
                    "SELECT COUNT(*) FROM t WHERE s IN ('zzz', 'Aardvark', 'nope')",
                    "SELECT id, s FROM t WHERE id IN (0, %d, %d) ORDER BY id" % (K - 1, n - 1),
                    "SELECT length(s) FROM t WHERE id = 5"]
            os.environ.update(FLOOR)
            for db_dir in (blk_dir, one_dir):
                db = Database.open(db_dir)
                for sql in sqls:
                    r = db.run(sql); got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
                    want = [tuple(x) for x in con.execute(sql).fetchall()]
                    assert got == want, (db_dir, sql, got, want)
            sz_b = sum(os.path.getsize(f) for f in glob.glob(blk_dir + '/t_*.wdb'))
            sz_1 = sum(os.path.getsize(f) for f in glob.glob(one_dir + '/t_*.wdb'))
            assert sz_b < 1.6 * sz_1, (sz_b, sz_1)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True)
