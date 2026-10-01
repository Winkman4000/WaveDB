"""THE CENSUS FROM THE DRESS (2026-10-01): Segment.raw_census counts rows per code straight from the
encoding (enc 5: nibble census + escapes; enc 10: run lengths + packed blocks) -- it must equal
np.bincount of the full decode for every column, with an odd row count (enc 5's pad nibble), and must
not build the decode. groupself answers GROUP BY K WHERE K <> x ORDER BY COUNT(*) DESC with no LIMIT."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
import wdb_engine
from wdb_engine import Segment

TMP = tempfile.gettempdir()


def _toy():
    rng = np.random.default_rng(41)
    n = 1_200_001                                                     # odd: enc 5's pad nibble
    hot = rng.integers(800, 2600, 15)
    width = np.where(rng.random(n) < 0.8, hot[np.minimum(rng.zipf(1.5, n), 15) - 1],
                     rng.integers(300, 2600, n)).astype(np.int64)     # ResolutionWidth's shape: tag 5
    adv = np.repeat(np.where(rng.random(n // 300 + 1) < 0.9, 0, rng.integers(1, 19, n // 300 + 1)), 300)[:n]
    adv = np.where(rng.random(n) < 0.002, rng.integers(0, 19, n), adv).astype(np.int64)   # runs + noise: tag 10
    flag = rng.integers(0, 3, n).astype(np.int64)
    d = os.path.join(TMP, 'cd_' + uuid.uuid4().hex[:8]); pq = d + '.parquet'
    os.makedirs(d)
    pd.DataFrame({'width': width, 'adv': adv, 'flag': flag}).to_parquet(pq, index=False)
    old = {k: os.environ.get(k) for k in _REAL}
    os.environ.update(_REAL)                         # the real load's rules: run.py opens mode 4 to toys
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'), stream=True)
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return d, pq


_REAL = {'WDB_SEQ_NARROW_OK': '0', 'WDB_SEQ_REPEATS_OK': '0'}


def test_census_from_the_dress_equals_the_decode():
    d, pq = _toy()
    try:
        P = os.path.join(d, 't_0.wdb')
        tags = {}
        for nm in ('width', 'adv', 'flag'):
            s = Segment(P)
            c = s.cols[nm]
            tags[nm] = c.get('code_enc')
            got = s.raw_census(nm, int(c['V']) + 3)
            if c.get('code_enc') in (5, 10):
                assert nm not in s._codes, ('the census decoded the column', nm)
            ref = np.bincount(np.asarray(Segment(P)._raw_codes(nm)), minlength=int(c['V']) + 3)
            assert got.size == ref.size and np.array_equal(got, ref), (nm, c.get('code_enc'))
            assert int(got.sum()) == int(s.N)
        assert 5 in tags.values() and 10 in tags.values(), tags        # both dresses were exercised
        old = wdb_engine._CENSUS[0]
        try:                                                          # the switch restores the decode
            wdb_engine._CENSUS[0] = False
            s = Segment(P)
            off = s.raw_census('width')
            assert 'width' in s._codes
            assert np.array_equal(off, np.bincount(np.asarray(s._raw_codes('width')), minlength=off.size))
        finally:
            wdb_engine._CENSUS[0] = old
    finally:
        shutil.rmtree(d, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass


def test_groupself_no_limit_equals_duck():
    import duckdb
    from wdb_db import Database
    d, pq = _toy()
    db_dir = d + '_db'
    try:
        import subprocess
        wdb = os.path.join(os.path.dirname(__file__), '..', 'bin', 'wdb')
        subprocess.run([sys.executable, wdb, 'load', db_dir, 't', pq], check=True, capture_output=True,
                       env=dict(os.environ, **_REAL))
        db = Database.open(db_dir)
        for sql in ('SELECT adv, COUNT(*) FROM t WHERE adv <> 0 GROUP BY adv ORDER BY COUNT(*) DESC',
                    'SELECT width, COUNT(*) AS c FROM t WHERE width <> 1000 GROUP BY width ORDER BY c DESC'):
            r = db.run(sql)
            rows = r[0] if isinstance(r, tuple) else r
            ref = duckdb.sql(sql.replace('FROM t', "FROM read_parquet('%s')" % pq)).fetchall()
            got = sorted((int(a), int(b)) for a, b in rows)
            assert got == sorted((int(a), int(b)) for a, b in ref), sql
            cnts = [int(b) for a, b in rows]
            assert cnts == sorted(cnts, reverse=True), sql                # ORDER BY count DESC held
    finally:
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(db_dir, ignore_errors=True)
        try: os.remove(pq)
        except OSError: pass
