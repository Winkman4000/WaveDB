"""Property-based fuzz round-trip: generate hundreds of diverse columns across every dtype
and shape, and assert lossless decode every time. This is the universal net -- it explores
the input space that hand-written tests can't, and (once mode-4 lands) it guards the new
codec against detector misfires, off-by-ones, and overflow on data we never hand-picked.

Seeded for reproducibility; each failure prints the exact spec so it can be reconstructed.
Cleans up its own temp files so /tmp doesn't bloat over hundreds of iterations."""
import sys, os, uuid, tempfile, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
from wdb_engine import Segment
from helpers import assert_lossless

TMP = tempfile.gettempdir()

def _rt_check(df, nm, spec):
    """Encode/decode df[nm], assert lossless, then delete the temp files. On failure, raise
    with the spec so it's reproducible."""
    tag = uuid.uuid4().hex[:8]
    pq = os.path.join(TMP, f'fz_{tag}.parquet'); wdb = os.path.join(TMP, f'fz_{tag}.wdb')
    try:
        df.to_parquet(pq, index=False)
        wdb_encode.encode(pq, wdb)
        assert_lossless(Segment(wdb), pq, nm)
    except Exception as e:
        raise AssertionError(f"fuzz lossless FAILED\n  spec={spec}\n  {type(e).__name__}: {e}\n"
                             f"{traceback.format_exc()}")
    finally:
        for f in (pq, wdb, wdb + '.tmp'):
            if os.path.exists(f):
                try: os.remove(f)
                except OSError: pass

def _gen_int(rng, n):
    shape = rng.choice(['seq','seq_stride','seq_gaps','shuffled_seq','rand_small',
                         'rand_wide','lowcard','single','two_run'])
    base = int(rng.integers(-10**12, 10**12))
    if shape == 'seq':            a = base + np.arange(n, dtype=np.int64)
    elif shape == 'seq_stride':   a = base + np.arange(n, dtype=np.int64) * int(rng.integers(1, 1000))
    elif shape == 'seq_gaps':
        stride = int(rng.integers(1, 10))
        a = base + np.cumsum(rng.integers(1, stride+3, size=n)).astype(np.int64)
    elif shape == 'shuffled_seq': a = base + np.arange(n, dtype=np.int64); rng.shuffle(a)
    elif shape == 'rand_small':   a = rng.integers(-50, 50, size=n).astype(np.int64)
    elif shape == 'rand_wide':    a = rng.integers(-10**15, 10**15, size=n).astype(np.int64)
    elif shape == 'lowcard':      a = rng.choice(rng.integers(-1000,1000,size=rng.integers(2,12)), size=n).astype(np.int64)
    elif shape == 'single':       a = np.full(n, base, dtype=np.int64)
    else:                         a = np.repeat(base + np.arange((n+1)//2, dtype=np.int64), 2)[:n]
    # optionally inject nulls (-> pandas Int64)
    if rng.random() < 0.35 and n > 0:
        s = pd.array(a, dtype='Int64')
        mask = rng.random(n) < rng.uniform(0.05, 0.5)
        s[mask] = pd.NA
        return s, shape + '+nulls'
    return a, shape

def _gen_float(rng, n):
    shape = rng.choice(['rand','lowcard','withnan','ints_as_float'])
    if shape == 'rand':          a = rng.uniform(-1e6, 1e6, size=n)
    elif shape == 'lowcard':     a = rng.choice([1.5,2.25,-7.5,0.0,3.125], size=n)
    elif shape == 'ints_as_float': a = rng.integers(-1000,1000,size=n).astype(np.float64)
    else:
        a = rng.uniform(-100,100,size=n); 
        if n: a[rng.random(n) < 0.3] = np.nan
    return a.astype(np.float64), shape

def _gen_str(rng, n):
    shape = rng.choice(['lowcard','highcard','unicode','withnull','empties'])
    if shape == 'lowcard':    pool = ['apple','banana','cherry','date']; a = list(rng.choice(pool, size=n))
    elif shape == 'highcard': a = [f'id_{int(x):09d}' for x in rng.integers(0,10**8,size=n)]
    elif shape == 'unicode':  pool = ['café','日本語','😀x','naïve','']; a = list(rng.choice(pool, size=n))
    elif shape == 'empties':  a = list(rng.choice(['','a',''], size=n))
    else:
        a = list(rng.choice(['a','b','c'], size=n).astype(object))
        for i in range(n):
            if rng.random() < 0.3: a[i] = None
    return pd.Series(a, dtype=object), shape

def _gen_dt(rng, n):
    base = np.datetime64('2015-06-01T00:00:00')
    shape = rng.choice(['seq_sec','rand_day','lowcard'])
    if shape == 'seq_sec':   offs = np.arange(n)
    elif shape == 'rand_day':offs = rng.integers(0, 30000, size=n)
    else:                    offs = rng.choice([0,1,2,3], size=n)
    unit = 's' if shape == 'seq_sec' else 'D'
    return pd.Series(base + offs.astype(f'timedelta64[{unit}]')), shape

def _one(rng, i):
    n = int(rng.integers(1, 4000))
    dt = rng.choice(['int','float','str','dt'], p=[0.5,0.2,0.2,0.1])  # bias to int (mode-4 turf)
    col, shape = {'int':_gen_int,'float':_gen_float,'str':_gen_str,'dt':_gen_dt}[dt](rng, n)
    df = pd.DataFrame({'x': col})
    _rt_check(df, 'x', f"#{i} dt={dt} shape={shape} n={n}")

def test_fuzz_roundtrip_300():
    rng = np.random.default_rng(20260603)
    for i in range(300):
        _one(rng, i)

def test_fuzz_roundtrip_int_heavy_150():
    # second seed, all-integer, biased to the sequence shapes mode-4 will target
    rng = np.random.default_rng(424242)
    for i in range(150):
        n = int(rng.integers(3, 5000))
        col, shape = _gen_int(rng, n)
        _rt_check(pd.DataFrame({'x': col}), 'x', f"int#{i} shape={shape} n={n}")
