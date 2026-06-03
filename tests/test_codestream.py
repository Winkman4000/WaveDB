"""Code-stream compression (WVDB4): the per-row code array of a mode-0/1/2 column is stored
as zstd of byte-aligned codes when that beats raw bit-packing (gated), else raw -- so clustered
/ skewed / sorted columns shrink hugely while incompressible ones pay only a 1-byte tag. Tag is
self-selected at encode and transparent at decode. Verified lossless (incl nulls) and query-
correct vs DuckDB. Mode 4 is disabled in the isolation tests so clustered columns land in mode
0/1/2 (otherwise the affine codec grabs them first); one test runs the REAL pipeline."""
import sys, os, uuid, tempfile, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql
from wdb_engine import Segment
from helpers import roundtrip

TMP = tempfile.gettempdir()
@contextlib.contextmanager
def no_mode4():
    orig = wdb_encode._try_seq
    wdb_encode._try_seq = lambda *a, **k: None
    try: yield
    finally: wdb_encode._try_seq = orig

def _enc(df):
    t = uuid.uuid4().hex[:8]; pq=f'{TMP}/cstest_{t}.parquet'; w=f'{TMP}/cstest_{t}.wdb'
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, w); return Segment(w), w, pq

def test_clustered_int_compresses_and_is_lossless():
    with no_mode4():
        col = np.minimum(np.random.default_rng(1).zipf(1.2, 1_000_000), 200000).astype(np.int64); col.sort()
        seg, w, _ = _enc(pd.DataFrame({'x': col}))
        raw = (seg.N * seg.cols['x']['bits'] + 7) // 8
        assert seg.cols['x']['code_enc'] == 1, "clustered codes should compress"
        assert os.path.getsize(w) * 5 < raw, "should be far smaller than raw bit-packed codes"
        assert np.array_equal(seg.values('x'), col)

def test_random_codes_stay_raw_and_lossless():
    with no_mode4():
        col = np.random.default_rng(2).integers(0, 200000, 1_000_000).astype(np.int64)
        seg, w, _ = _enc(pd.DataFrame({'x': col}))
        assert seg.cols['x']['code_enc'] == 0, "incompressible codes must stay raw (gate)"
        assert np.array_equal(seg.values('x'), col)

def test_clustered_string_compresses_and_is_lossless():
    with no_mode4():
        s = np.random.default_rng(3).choice(['ok','retry','fail','queued','cancel','timeout'],
                                             1_000_000, p=[.7,.1,.05,.08,.04,.03]); s.sort()
        seg, w, _ = _enc(pd.DataFrame({'s': s}))
        assert seg.cols['s']['code_enc'] == 1
        assert np.array_equal(np.array([x.decode() for x in seg.values('s')]), s)

def test_code_width_variants_lossless():
    # exercises 1/2/4-byte code widths: low-, mid-, high-cardinality sorted columns
    with no_mode4():
        for card in (200, 40_000, 300_000):
            col = np.minimum(np.random.default_rng(card).zipf(1.15, 600_000), card).astype(np.int64); col.sort()
            seg, w, _ = _enc(pd.DataFrame({'x': col}))
            assert np.array_equal(seg.values('x'), col), f"card={card}"

def test_nullable_clustered_lossless():
    with no_mode4():
        v = pd.array(np.minimum(np.random.default_rng(4).zipf(1.3, 500_000), 50000), dtype='Int64')
        v[::997] = pd.NA
        df = pd.DataFrame({'n': v}).sort_values('n', na_position='last').reset_index(drop=True)
        seg, w, _ = _enc(df)
        out = seg.values('n'); exp = df['n'].tolist()
        nul = lambda x: x is None or x is pd.NA or (isinstance(x, float) and np.isnan(x))
        assert all((nul(a) and nul(b)) or (not nul(a) and not nul(b) and int(a)==int(b))
                   for a, b in zip(exp, out))

def test_queries_match_oracle_on_compressed_column():
    with no_mode4():
        n = 200_000
        cat = np.random.default_rng(5).choice([f"c{i}" for i in range(2000)], n); cat.sort()  # clustered -> compress
        amt = np.random.default_rng(6).integers(0, 100, n).astype(np.int64)
        df = pd.DataFrame({'cat': cat, 'amt': amt})
        seg, w, pq = _enc(df)
        assert seg.cols['cat']['code_enc'] == 1
        con = duckdb.connect()
        for sql in ["SELECT COUNT(*) FROM TBL WHERE cat = 'c1000'",
                    "SELECT cat, COUNT(*) FROM TBL GROUP BY cat",
                    "SELECT SUM(amt) FROM TBL WHERE cat = 'c500'"]:
            duck = sorted(con.execute(sql.replace('TBL', f"'{pq}'")).fetchall())
            rows, _ = wdb_sql.execute(seg, sql.replace('TBL', 'tbl'))
            assert sorted(rows) == duck, sql

def test_real_pipeline_runlength_below_mode4_threshold():
    # mode 4 ON (real pipeline): run-length 4 -> 75% adjacent repeats < 80% -> mode 4 declines,
    # column lands in mode 2 and its run-structured codes get code-stream compressed.
    base = np.random.default_rng(7).integers(0, 80_000, 250_000)
    col = np.repeat(base, 4).astype(np.int64)[:1_000_000]
    seg, w, _ = _enc(pd.DataFrame({'x': col}))
    assert seg.cols['x']['mode'] in (0, 2) and seg.cols['x'].get('code_enc') == 1
    assert np.array_equal(seg.values('x'), col)
