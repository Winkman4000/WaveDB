"""Bytes per column of two segments (columns are written one after another, so a column's bytes run
from its first offset to the next column's). Prints the columns whose size moved most between them.
Usage: PYTHONPATH=src python bench/col_sizes.py SEG_A SEG_B [top]
"""
import sys, os
import wdb_engine


def spans(path):
    s = wdb_engine.Segment(path)
    starts = {}
    for c, m in s.cols.items():
        offs = [int(v) for k, v in m.items() if isinstance(v, (int,)) and not isinstance(v, bool)
                and (k.endswith('off') or k.endswith('start') or k in ('blob', 'i2base', 'pXdir', 'pXpay'))
                and 0 < int(v) < os.path.getsize(path)]
        if offs:
            starts[c] = min(offs)
    order = sorted(starts, key=starts.get)
    end = os.path.getsize(path)
    out = {}
    for i, c in enumerate(order):
        out[c] = (starts[order[i + 1]] if i + 1 < len(order) else end) - starts[c]
    return out, s


a, sa = spans(sys.argv[1]); b, sb = spans(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rows = sorted(set(a) & set(b), key=lambda c: a[c] - b[c], reverse=True)
tot = sum(a[c] - b[c] for c in rows)
print('total over %d columns: A %.2f GB, B %.2f GB, A - B = %.0f MB' % (len(rows), sum(a.values()) / 1e9, sum(b.values()) / 1e9, tot / 1e6))
print('%-22s %9s %9s %9s  %s' % ('column', 'A MB', 'B MB', 'saved', 'encoding A -> B, distinct values'))
for c in rows[:top] + rows[-5:]:
    ma, mb = sa.cols[c], sb.cols[c]
    print('%-22s %9.1f %9.1f %9.1f  enc %s/%s -> %s/%s, V %s' % (c, a[c] / 1e6, b[c] / 1e6, (a[c] - b[c]) / 1e6,
          ma.get('mode'), ma.get('code_enc'), mb.get('mode'), mb.get('code_enc'), ma.get('V')))
