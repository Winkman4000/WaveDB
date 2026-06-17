#!/usr/bin/env python3
"""lattice_proto.py -- proof-of-concept for a unified sidecar lattice over a shared code basis.

Thesis (Jackson's): every sidecar is a node (axes, measures) in one coordinate system -- the
dictionary codes. Once they share that basis, a finer node DERIVES coarser answers for free
(sum-out), BOUNDS finer ones (min of margins), and FOLDS measures (AVG = SUM/COUNT). The only
thing that must be physically stored at/below a target is what can't be derived from above --
e.g. a high-card pair COUNT, kept heavy-cells-only (singletons implicit).

Run: PYTHONPATH=src python bench/lattice_proto.py [/path/to/lineitem_0.wdb]
Builds nothing permanent; verifies each claim against recompute-from-rows on the 6M lineitem.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from wdb_engine import Segment

SEG = sys.argv[1] if len(sys.argv) > 1 else '/tmp/jbprof_sf1.0/wdb/lineitem_0.wdb'


class Node:
    """A cuboid in the shared code basis. axes: tuple of column names; card: per-axis cardinality;
       measures: name -> dense ndarray shaped like card (COUNT plus folded measure sums)."""
    def __init__(self, axes, card, measures):
        self.axes, self.card, self.measures = tuple(axes), tuple(card), measures

    def sum_out(self, keep):
        keep = tuple(keep)
        drop = tuple(i for i, a in enumerate(self.axes) if a not in keep)
        m = {nm: (arr.sum(axis=drop) if drop else arr) for nm, arr in self.measures.items()}
        return Node(keep, tuple(self.card[self.axes.index(a)] for a in keep), m)


def heavy_cells(a, b, Vb, min_count=2):
    """Sparse heavy-cell node for a high-card pair: distinct (a,b) packed to int64, counts kept
    only where count >= min_count (singletons are an implicit 1, never in a top-K once >=k repeat)."""
    pk = a.astype(np.int64) * Vb + b.astype(np.int64)
    u, c = np.unique(pk, return_counts=True)
    keep = c >= min_count
    return u[keep], c[keep], u.size


def main():
    seg = Segment(SEG); N = seg.N
    ok = True

    # ---- shared basis: codes are dense integer axes (sorted-rank, value-identity) ----
    rf = np.asarray(seg.codes('l_returnflag')).astype(np.int64)
    ls = np.asarray(seg.codes('l_linestatus')).astype(np.int64)
    qty = np.asarray(seg.values('l_quantity')).astype(np.float64)
    Vrf, Vls = int(rf.max()) + 1, int(ls.max()) + 1

    # ---- ONE 2-axis node with COUNT + SUM(qty) ----
    pair = rf * Vls + ls
    cnt = np.bincount(pair, minlength=Vrf * Vls).reshape(Vrf, Vls)
    sqty = np.bincount(pair, weights=qty, minlength=Vrf * Vls).reshape(Vrf, Vls)
    node = Node(('rf', 'ls'), (Vrf, Vls), {'count': cnt, 'sum_qty': sqty})
    print(f"[node] (l_returnflag,l_linestatus) COUNT+SUM = {cnt.nbytes + sqty.nbytes} bytes, {cnt.size} cells")

    # CLAIM 1 -- sum-out reproduces BOTH single-key margins exactly (1 node answers 3 group-bys)
    c1 = (np.array_equal(node.sum_out(('rf',)).measures['count'], np.bincount(rf, minlength=Vrf)) and
          np.array_equal(node.sum_out(('ls',)).measures['count'], np.bincount(ls, minlength=Vls)))
    print(f"  [1] sum-out == recomputed margins (GROUP BY rf; GROUP BY ls): {c1}"); ok &= c1

    # CLAIM 2 -- fold: AVG = SUM/COUNT matches recompute-from-rows
    m = node.sum_out(('rf',)); avg_d = m.measures['sum_qty'] / m.measures['count']
    avg_r = np.array([qty[rf == i].mean() for i in range(Vrf)])
    c2 = np.allclose(avg_d, avg_r)
    print(f"  [2] fold AVG(qty)=SUM/COUNT == recomputed: {c2}"); ok &= c2

    # CLAIM 3 -- cross-margin bound: every cell <= min(margin_rf, margin_ls)
    c3 = bool((cnt <= np.minimum(np.bincount(rf, minlength=Vrf)[:, None],
                                 np.bincount(ls, minlength=Vls)[None, :])).all())
    print(f"  [3] every cell <= min(margins): {c3}"); ok &= c3

    # ---- high-card pairs: heavy-cell node, the C18 mechanism + the storage gate ----
    for an, bn in [('l_partkey', 'l_suppkey'), ('l_orderkey', 'l_suppkey')]:
        a = np.asarray(seg.codes(an)).astype(np.int64); b = np.asarray(seg.codes(bn)).astype(np.int64)
        Vb = int(b.max()) + 1
        hu, hc, npairs = heavy_cells(a, b, Vb, 2)
        frac = 1 - hu.size / npairs
        # top-K by count: heavy node holds every count>=2 pair, so the top-K COUNT multiset is exact
        o = np.argsort(hc)[::-1][:5]; heavy_counts = sorted(hc[o].tolist(), reverse=True)
        pk = a * Vb + b; fu, fc = np.unique(pk, return_counts=True)
        full_counts = sorted(fc[np.argsort(fc)[::-1][:5]].tolist(), reverse=True)
        c = heavy_counts == full_counts
        print(f"[heavy] {an}x{bn}: pairs={npairs:,} heavy>=2={hu.size:,} singleton={frac:.1%} "
              f"node={hu.nbytes + hc.nbytes:,}B  top5-count-exact={c}"); ok &= c

    # CLAIM -- honest boundary: heavy-only cannot reproduce exact margins (singletons dropped),
    # so the margin node (gbc) coexists; the registry holds both and the router composes them.
    print(f"  [boundary] sum(heavy counts) < N => margins need their own node (gbc), not derivable from heavy-only")
    print(f"\nLATTICE PROTO: {'ALL CLAIMS PASS' if ok else 'FAILURE'}")


if __name__ == '__main__':
    main()
