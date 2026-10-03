"""THE JOB FLOOR (2026-10-03): the semi-join organ with sidecars off builds nothing it cannot keep. The new
kernels equal their references -- the extreme row of an inline column (bytewise, prefixes first), = / IN on
the inline stream, the one-pass cut (full and over live rows, NULL keys), the int32 gather, the parallel
unpack at every width; and through SQL, MIN/MAX of inline strings, = / <> / IN / LIKE on inline columns and a
nullable integer join key answer as DuckDB does, with no string rank or road born."""
import sys, os, uuid, tempfile, shutil, subprocess, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd

TMP = tempfile.gettempdir()


def _stream(strs):
    b = [s if isinstance(s, bytes) else s.encode() for s in strs]
    off = np.zeros(len(b) + 1, np.int64); np.cumsum([len(x) for x in b], out=off[1:])
    return np.frombuffer(b''.join(b) or b'\x00', dtype=np.uint8), off, b


def test_inline_extreme_and_equality_kernels():
    import wdb_semijoin as S, wdb_kernels as K
    rng = np.random.default_rng(81)
    alpha = [b'', b'a', b'ab', b'abc', b'b', b'\xff', b'a\xff', b'\x00', b'ab\x00', b'ba']
    rand = [bytes(rng.integers(0, 256, int(rng.integers(0, 6))).astype(np.uint8)) for _ in range(3000)]
    blob, off, b = _stream(alpha + rand)
    for trial in range(40):
        rows = np.unique(rng.integers(0, len(b), int(rng.integers(1, 400)))).astype(np.int64)
        rng.shuffle(rows)
        for want_max in (False, True):
            j = S._inline_extreme(blob, off, rows, want_max)
            ref = (max if want_max else min)(b[r] for r in rows)
            assert b[j] == ref, (trial, want_max, b[j], ref)
    for lits in ([b'ab'], [b''], [b'a', b'\xff', b'zz'], [b'abc', b'ab'], [rand[7], rand[8]]):
        lb, lo, _ = _stream(lits)
        out = np.empty(len(b), np.bool_)
        K.pinline_eq_any(blob, off, lb, lo, out)
        assert np.array_equal(out, np.array([x in set(lits) for x in b])), lits


def test_like_token_kernel_equals_regex():
    import re, itertools
    import wdb_kernels as K, wdb_sql
    rng = np.random.default_rng(84)
    strs = [''.join(rng.choice(list('ab'), int(rng.integers(0, 7)))) for _ in range(400)] + ['', 'a', 'b', 'ab', 'ba', 'aab']
    blob, off, b = _stream(strs)
    rows = np.unique(rng.integers(0, len(b), 120)).astype(np.int64)
    pats = [''.join(p) for L in range(0, 6) for p in itertools.product('ab%', repeat=L)]
    for pat in pats:
        tb, to, a0, a1 = wdb_sql._like_tok_parts(pat)
        rx = re.compile('^' + ''.join('.*' if ch == '%' else re.escape(ch) for ch in pat) + '$', re.S)
        ref = np.array([rx.match(s.decode()) is not None for s in b])
        out = np.empty(len(b), np.bool_)
        K.plike_tok(blob, off, tb, to, a0, a1, out)
        assert np.array_equal(out, ref), (pat, [s for s, o, r in zip(b, out, ref) if o != r][:5])
        out2 = np.empty(rows.size, np.bool_)
        K.plike_tok_rows(blob, off, rows, tb, to, a0, a1, out2)
        assert np.array_equal(out2, ref[rows]), pat
    assert wdb_sql._like_tok_parts('a_b') is None and wdb_sql._like_tok_parts('a\\%') is None


def test_cut_gather_and_unpack():
    import wdb_semijoin as S, wdb_engine
    rng = np.random.default_rng(82)
    for n, mx in ((0, 10), (1, 1), (1000, 50), (300_001, 70_000)):
        keys = rng.integers(-1, mx, n).astype(np.int32)
        live = np.unique(rng.integers(0, mx, max(1, mx // 7)))
        lut = np.zeros(mx + 2, bool); lut[live] = True
        ref_full = np.flatnonzero((keys >= 0) & lut[np.where(keys < 0, mx + 1, keys)])
        assert np.array_equal(S._cut_rows(keys, S._NOIDX, lut, True), ref_full), n
        idx = np.unique(rng.integers(0, max(n, 1), n // 3)).astype(np.int64) if n else S._NOIDX
        ref_idx = idx[(keys[idx] >= 0) & lut[np.where(keys[idx] < 0, mx + 1, keys[idx])]] if n else idx
        assert np.array_equal(S._cut_rows(keys, idx, lut, False), ref_idx), n
        vals = rng.integers(-1, 1 << 30, mx + 1).astype(np.int32)
        codes = rng.integers(0, mx + 1, n).astype(np.uint32)
        assert np.array_equal(S._gather_i32(vals, codes), vals[codes]), n

    class _Buf:                                   # the unpack reads only self.buf
        pass
    for bits in (1, 3, 7, 9, 13, 17, 21, 22, 24, 25):
        n = 70_000 + bits * 13
        vals = rng.integers(0, 1 << bits, n, dtype=np.uint64)
        m = ((vals[:, None] >> np.arange(bits - 1, -1, -1, dtype=np.uint64)) & 1).astype(np.uint8)
        x = _Buf(); x.buf = np.packbits(m.reshape(-1))       # no tail padding: the last window reads past it
        for lo, hi in ((0, n), (5, n), (65_537, n), (3, 3 + 66_000)):
            got = wdb_engine.Segment._bitunpack(x, 0, lo, hi, bits)
            assert np.array_equal(got.astype(np.uint64), vals[lo:hi]), (bits, lo, hi)


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update(kv)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _frames():
    rng = np.random.default_rng(83)
    nt = 260_000
    words = np.array(['alpha', 'beta', 'gamma', 'delta', 'Downey', 'Robert', 'Shrek', 'Queen'], dtype=object)
    title = np.array(['%s %s %07d' % (words[i % 8], words[(i * 7) % 8], int(v)) for i, v in
                      enumerate(rng.permutation(nt))], dtype=object)
    title[[11, 222, 3333]] = ['Shrek 2', 'Queen', '']
    t = pd.DataFrame({'id': np.arange(1, nt + 1, dtype=np.int64), 'title': title,
                      'yr': rng.integers(1950, 2020, nt)})
    nc = 900_000
    note = np.array(['n%06d %s%s' % (int(v), words[int(v) % 8], words[int(v) % 5]) for v in
                     rng.integers(0, 10 ** 6, nc)], dtype=object)
    role = rng.integers(1, 40_000, nc).astype(float)
    role[rng.random(nc) < 0.3] = np.nan                      # a nullable integer join key
    c = pd.DataFrame({'movie_id': rng.integers(1, nt + 1, nc), 'note': note, 'role_id': pd.array(role, dtype='Int64')})
    r = pd.DataFrame({'id': np.arange(1, 40_001, dtype=np.int64), 'role': ['r%d' % (i % 9) for i in range(40_000)]})
    return {'t': t, 'c': c, 'r': r}


def test_semijoin_through_sql_equals_duck():
    import duckdb
    from wdb_db import Database
    d = os.path.join(TMP, 'jf_' + uuid.uuid4().hex[:8]); db_dir = d + '_db'
    os.makedirs(d)
    floor = dict(WDB_LOAD_ANSWERS='0', WDB_SIDECARS='0', WDB_QMEM_STRICT='1')
    try:
        con = duckdb.connect()
        wdb = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
        for name, df in _frames().items():
            pq = os.path.join(d, name + '.parquet'); df.to_parquet(pq, index=False)
            subprocess.run([sys.executable, wdb, 'load', db_dir, name, pq], check=True, capture_output=True,
                           env=dict(os.environ, **floor))
            con.execute("CREATE VIEW %s AS SELECT * FROM read_parquet('%s')" % (name, pq))
        sqls = [
            "SELECT MIN(t.title), MAX(t.title), MIN(c.note) FROM t, c WHERE t.id = c.movie_id AND c.note LIKE '%Downey%Robert%'",
            "SELECT MIN(t.title), MAX(c.note) FROM t, c WHERE t.id = c.movie_id AND t.yr BETWEEN 1990 AND 1995 AND c.note LIKE '%gamma%'",
            "SELECT MIN(c.note), MAX(t.title) FROM t, c WHERE t.id = c.movie_id AND t.title = 'Shrek 2'",
            "SELECT MIN(c.note), MAX(t.yr) FROM t, c WHERE t.id = c.movie_id AND t.title IN ('Queen', 'Shrek 2', 'nope')",
            "SELECT MIN(t.title), COUNT(*) FROM t, c WHERE t.id = c.movie_id AND t.title <> 'Queen' AND c.note LIKE '%deltaa%'",
            "SELECT MIN(t.title), MIN(r.role) FROM t, c, r WHERE t.id = c.movie_id AND c.role_id = r.id AND r.role = 'r3' AND t.yr > 2015",
            "SELECT MIN(t.title) FROM t, c WHERE t.id = c.movie_id AND c.note NOT LIKE '%alpha%' AND t.title LIKE '%Shrek%Queen%'",
            "SELECT MAX(t.title) FROM t, c WHERE t.id = c.movie_id AND t.title = ''",
            # deferred (no kernel says '_' or ILIKE): asked of the survivors, few and many
            "SELECT MIN(t.title), MAX(c.note) FROM t, c WHERE t.id = c.movie_id AND t.title LIKE 'S_rek%' AND t.yr = 1990",
            "SELECT MIN(t.title), MIN(c.note) FROM t, c WHERE t.id = c.movie_id AND t.title ILIKE '%shrek%' AND c.note LIKE 'n00%'",
            "SELECT MIN(c.note), COUNT(*) FROM t, c WHERE t.id = c.movie_id AND (c.note LIKE 'n0001%' OR t.yr = 1999) AND t.title NOT LIKE 'Queen _eta%'",
            "SELECT MIN(t.title), MAX(t.title) FROM t, c, r WHERE t.id = c.movie_id AND c.role_id = r.id AND r.role = 'r1' AND t.title LIKE '%a_p%'",
            "SELECT MIN(t.title) FROM t, c WHERE t.id = c.movie_id AND t.title LIKE 'Shrek%' AND c.note LIKE '%Shrek%gamma%'",
        ]
        with _env(**floor):
            db = Database.open(db_dir)
            for sql in sqls:
                r = db.run(sql); got = [tuple(x) for x in (r[0] if isinstance(r, tuple) else r)]
                ref = [tuple(x) for x in con.execute(sql).fetchall()]
                assert got == ref, (sql, got, ref)
            segs = [ent[1] for ent in db._seg_cache.values()]
            seg = next(s for s in segs if 'title' in s.cols)
            assert seg.cols['title'].get('mode') == 5, seg.cols['title'].get('mode')      # the inline column ran
            cseg = next(s for s in segs if 'role_id' in s.cols)
            assert cseg.cols['role_id'].get('has_null') and cseg.cols['role_id'].get('mode') in (0, 1, 2)
        born = [f for f in os.listdir(db_dir) if f.endswith(('.srank.npy', '.npy', '.hdr.npy'))]
        assert not born, born                                                            # nothing kept, nothing built to disk
    finally:
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
