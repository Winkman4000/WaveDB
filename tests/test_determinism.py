"""Determinism & idempotency. Compaction reconstructs a union via values() and re-encodes
it; mode-4 will re-run its detector on that union. For that to be safe, encode must be
byte-deterministic, decode repeatable, and re-encoding decoded data a fixpoint. Pin all three
now so the migration can't quietly introduce nondeterminism (e.g. from the parallel encode)."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
import wdb_encode
from wdb_engine import Segment
from helpers import roundtrip, recon_list, orig_list

TMP = tempfile.gettempdir()

def _mixed_df(n=3000):
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        'lowint':  rng.integers(0, 8, n).astype(np.int64),
        'hiint':   np.arange(n, dtype=np.int64),                       # mode 2
        'lostr':   rng.choice(['a','bb','ccc'], n),
        'histr':   [f'k_{i:07d}' for i in range(n)],                   # mode 1
        'flt':     np.round(rng.uniform(-100,100,n), 3),
        'dt':      np.datetime64('2021-01-01') + rng.integers(0,5000,n).astype('timedelta64[s]'),
    })

def _encode_to(df, tag):
    pq = os.path.join(TMP, f'det_{tag}.parquet'); wdb = os.path.join(TMP, f'det_{tag}.wdb')
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, wdb)
    return pq, wdb

def test_encode_byte_deterministic():
    df = _mixed_df()
    _, w1 = _encode_to(df, uuid.uuid4().hex[:8])
    _, w2 = _encode_to(df, uuid.uuid4().hex[:8])
    b1 = open(w1,'rb').read(); b2 = open(w2,'rb').read()
    assert b1 == b2, f"encode not byte-deterministic: {len(b1)} vs {len(b2)} bytes, differ"

def test_decode_repeatable():
    seg, pq = roundtrip(_mixed_df())
    for nm in seg.cols:
        v1 = seg.values(nm); v2 = seg.values(nm)
        c1 = seg.codes(nm);  c2 = seg.codes(nm)
        assert np.array_equal(np.asarray(v1, object), np.asarray(v2, object)), f"values({nm}) not repeatable"
        assert np.array_equal(c1, c2), f"codes({nm}) not repeatable"

def test_reencode_is_fixpoint():
    # encode -> decode -> feed decoded values back -> re-encode -> decode == original.
    # This is the compaction operation; it must be a fixpoint (clean, non-null columns).
    df = pd.DataFrame({
        'hiint': np.arange(4000, dtype=np.int64),
        'lowint': (np.arange(4000) % 5).astype(np.int64),
        'lostr': (['x','yy','zzz','w']*1000),
        'flt':   np.round(np.linspace(-50,50,4000), 4),
    })
    seg1, pq1 = roundtrip(df)
    df2 = pd.DataFrame({nm: seg1.values(nm) for nm in df.columns})
    # normalize bytes->str for the string column so parquet stores the same logical type
    df2['lostr'] = [b.decode() if isinstance(b,(bytes,bytearray)) else b for b in df2['lostr']]
    seg2, pq2 = roundtrip(df2)
    for nm in df.columns:
        assert recon_list(seg1, nm) == recon_list(seg2, nm), f"re-encode not a fixpoint for {nm}"
    # and byte-stable on the second vs third pass (now both come from decoded data)
    df3 = pd.DataFrame({nm: seg2.values(nm) for nm in df.columns})
    df3['lostr'] = [b.decode() if isinstance(b,(bytes,bytearray)) else b for b in df3['lostr']]
    seg3, _ = roundtrip(df3)
    for nm in df.columns:
        assert recon_list(seg2, nm) == recon_list(seg3, nm), f"second re-encode drifted for {nm}"

def test_high_card_sequence_byte_deterministic():
    # the mode-4-relevant shape specifically: byte-identical across encodes
    df = pd.DataFrame({'id': 1_000_000 + np.arange(80000, dtype=np.int64)})
    _, w1 = _encode_to(df, uuid.uuid4().hex[:8])
    _, w2 = _encode_to(df, uuid.uuid4().hex[:8])
    assert open(w1,'rb').read() == open(w2,'rb').read(), "sequence encode not byte-deterministic"
