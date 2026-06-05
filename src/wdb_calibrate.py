#!/usr/bin/env python3
"""
wdb_calibrate -- measure THIS host's analog constants, once.

Theory: S = work x rate(locality). The rate is set by the medium, so it must be
measured per machine, not assumed. This writes:

    seq_rate_ns       sequential read cost (ns/element)
    rand_rate_ns      random-gather cost (ns/element)
    locality_penalty  rand/seq -- a scattered plan must cut work by MORE than this
                      to beat a sequential scan (else: scan, or cluster)
    lut_cliff         max dict/LUT entries that stay cache-resident -> code-LUT
                      eligibility (caps LUT_MAX_CARD)

to ~/.wavedb/calibration.json. The planner reads from it; nothing is hand-set.
"""
import os, json, time, math
import numpy as np

CALIB_PATH = os.path.expanduser('~/.wavedb/calibration.json')


def _best(fn, n=20):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


def _kernels():
    """numba kernels: clean rate measurement with no numpy temporaries.
    Returns (seq_sum, gather_sum_f64, gather_sum_u8) or None if numba absent."""
    try:
        from numba import njit
    except Exception:
        return None

    @njit
    def seq(a, n):
        s = 0.0
        for i in range(n):
            s += a[i]
        return s

    @njit
    def gather(a, idx, n):
        s = 0.0
        for i in range(n):
            s += a[idx[i]]
        return s

    @njit
    def gather_u8(a, idx, n):
        s = 0
        for i in range(n):
            s += a[idx[i]]
        return s

    return seq, gather, gather_u8


def _lut_cliff(gather_u8, rng, probe_n=4_000_000):
    """Largest bool-LUT (uint8) size whose random-gather rate stays within 1.6x of
    the smallest -- i.e. still effectively cache-resident. Caps LUT_MAX_CARD."""
    probe = rng.integers(0, 1 << 24, size=probe_n).astype(np.int64)
    base = None
    cliff = 1 << 12
    for bits in range(10, 25):
        V = 1 << bits
        arr = (rng.random(V) < 0.5).astype(np.uint8)
        idx = (probe % V)
        gather_u8(arr, idx, probe_n)                      # warm
        t = _best(lambda arr=arr, idx=idx: gather_u8(arr, idx, probe_n), n=7)
        rate = t / probe_n
        if base is None:
            base = rate
        if rate <= 1.6 * base:
            cliff = V
        else:
            break
    return int(cliff)


def _numpy_fallback(a, ridx, n):
    """If numba is unavailable: coarse rates via numpy (gather allocates a temp,
    so the penalty is under-measured -- flagged in the output)."""
    t_seq = _best(lambda: a.sum())
    t_rand = _best(lambda: a[ridx].sum())
    return t_seq / n, t_rand / n, None


def calibrate(n_elem=16_000_000, seed=0, measure_cliff=True):
    rng = np.random.default_rng(seed)
    a = rng.random(n_elem)
    ridx = rng.integers(0, n_elem, size=n_elem).astype(np.int64)
    k = _kernels()
    engine = 'numba'
    if k is None:
        seq_rate, rand_rate, cliff = _numpy_fallback(a, ridx, n_elem)
        engine = 'numpy_fallback'
    else:
        seq, gather, gather_u8 = k
        seq(a, n_elem); gather(a, ridx, n_elem)          # compile
        seq_rate = _best(lambda: seq(a, n_elem)) / n_elem
        rand_rate = _best(lambda: gather(a, ridx, n_elem)) / n_elem
        cliff = _lut_cliff(gather_u8, rng) if measure_cliff else 65536
    penalty = (rand_rate / seq_rate) if seq_rate else None
    return {
        'seq_rate_ns': round(seq_rate * 1e9, 4),
        'rand_rate_ns': round(rand_rate * 1e9, 4),
        'locality_penalty': round(penalty, 3) if penalty else None,
        'lut_cliff': cliff if cliff is not None else 65536,
        'n_cores': os.cpu_count(),
        'n_elem': n_elem,
        'engine': engine,
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def save_calibration(d, path=CALIB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    return path


def load_calibration(path=CALIB_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get(path=CALIB_PATH):
    """Lazy: load cached constants, else calibrate-and-save."""
    d = load_calibration(path)
    if d is None:
        d = calibrate(); save_calibration(d, path)
    return d


if __name__ == '__main__':
    d = calibrate()
    p = save_calibration(d)
    print("WaveDB host calibration ->", p)
    for k_, v in d.items():
        print(f"  {k_:18s} {v}")
    if d.get('locality_penalty'):
        print(f"\nDECISION RULE: a scattered plan must cut work by > "
              f"{d['locality_penalty']}x to beat a sequential scan; else scan or cluster.")
