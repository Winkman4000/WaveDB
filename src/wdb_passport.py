"""wdb_passport -- every column publishes a card of self-measured facts.

The router's vocabulary. v1 derives everything from headers already on disk
(zero new storage); lazy facts are measured once per open segment on first
ask and cached in memory only. No query-instance data ever."""
import numpy as np


def card(seg, nm):
    """The column's passport: cheap header facts, always available."""
    c = seg.cols.get(nm)
    if c is None:
        return None
    p = {
        'name': nm, 'dt': c.get('dt'), 'N': int(seg.N),
        'mode': c.get('mode'), 'code_enc': c.get('code_enc'),
        'cwidth': c.get('cwidth'), 'BR': c.get('BR'),
        'V': int(c.get('V', 0)), 'has_null': bool(c.get('has_null')),
        'dict_chunked': bool(c.get('chunked')) or c.get('i2ch') is not None,
        'front_coded': c.get('R') is not None,
        'stair': bool(c.get('stair')),
    }
    if c.get('code_enc') == 8:
        p['majority_share'] = float(c.get('e8_share', 0.0)) or None
    if c.get('code_enc') == 5:
        p['hot_codes'] = int(np.asarray(c.get('e5hot')).size) if c.get('e5hot') is not None else None
    p['charlens_ready'] = c.get('charlens') is not None or c.get('bytelens') is not None
    return p


def coverage(seg, nm, ks=(1, 100, 10000, 1000000)):
    """LAZY fact: head-coverage curve -- what share of rows the top-k codes
    carry. One full code read + bincount on first ask; cached in memory."""
    c = seg.cols.get(nm)
    if c is None:
        return None
    got = c.get('_pp_cov')
    if got is not None:
        return got
    codes = np.asarray(seg._raw_codes(nm))
    cnt = np.bincount(codes, minlength=int(c['V']))
    sc = np.sort(cnt)[::-1]
    cum = np.cumsum(sc, dtype=np.int64)
    tot = int(cum[-1]) if cum.size else 1
    cov = {int(k): float(cum[min(k, cum.size) - 1] / tot) for k in ks}
    c['_pp_cov'] = cov
    return cov
