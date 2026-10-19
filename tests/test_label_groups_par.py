"""THE PARALLEL LABEL GROUPS (2026-10-19): label_groups_par puts two labels in one group exactly when their
bytes are equal -- with real hashes, and with hashes forced to collide (every label one hash; a few hashes)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_kernels as WK


def _labels(n, seed):
    rng = np.random.default_rng(seed)
    alpha = np.frombuffer(b'abc./-xyz\xd0\xbf', np.uint8)
    labs = [bytes(alpha[rng.integers(0, alpha.size, int(rng.integers(0, 6)))]) for _ in range(n)]
    blob = np.frombuffer(b''.join(labs) or b'\x00', np.uint8).copy()
    offs = np.zeros(n + 1, np.int64); np.cumsum([len(x) for x in labs], out=offs[1:])
    return labs, blob, offs


def _check(labs, blob, offs, h):
    n = len(labs)
    gid = np.empty(n, np.int64); rep = np.empty(n, np.int64)
    G = int(WK.label_groups_par(blob, offs, h, gid, rep))
    assert G == len(set(labs)), (G, len(set(labs)))
    assert n == 0 or (gid.min() >= 0 and gid.max() == G - 1)
    first = {}
    for r, lab in enumerate(labs):
        g = first.setdefault(lab, gid[r])
        assert gid[r] == g, ('one label, two groups', lab)
    assert len(set(first.values())) == G, 'two labels, one group'
    for g in range(G):
        assert labs[rep[g]] in first and first[labs[rep[g]]] == g


def test_label_groups_par_real_hashes():
    labs, blob, offs = _labels(200_000, 1)
    h = np.empty(len(labs), np.uint64); WK.label_hash(blob, offs, h)
    _check(labs, blob, offs, h)


def test_label_groups_par_forced_collisions():
    labs, blob, offs = _labels(3000, 2)
    _check(labs, blob, offs, np.full(len(labs), np.uint64(12345), np.uint64))          # one hash for all
    h = np.empty(len(labs), np.uint64); WK.label_hash(blob, offs, h)
    _check(labs, blob, offs, h & np.uint64(7))                      # 8 hashes, one bucket (a hash is the label's)
    _check(labs, blob, offs, (h % np.uint64(3)) << np.uint64(60))  # 3 hashes in 3 buckets
    _check(labs[:0], blob[:1], offs[:1], np.empty(0, np.uint64))                         # no label
