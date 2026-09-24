"""THE CODE AUTOPSY (2026-09-24): the per-row code sections the cold board decodes whole.

Columns are chosen by the board itself: the scatter census (scatter_i2.jsonl) gives the cold time
spent in _raw_codes / _raw_codes_range per column; the heaviest are dissected. Per column:
  - the bill: a full decode cold (the file evicted, a fresh Segment) -- time and bytes pulled
    from storage -- against the same decode warm (CPU only). Cold minus warm is the waiting.
  - bits per row: stored (bytes read x 8 / rows), plain packing (log2 of the dictionary size),
    and the information the codes carry: H0, the entropy of the code distribution; and the share
    of rows whose code equals the row before (runs).
  - for zstd-framed codes (enc 3): the autopsy of real frames -- literals in place, copy rules,
    how far into a frame a value is ready.

Usage: PYTHONPATH=src python bench/code_autopsy.py DB_DIR SCATTER_JSONL TRACE_BIN [ncols]
"""
import sys, os, glob, json, time, subprocess, collections
import numpy as np
import wdb_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zstd_autopsy as ZA


def resident(f):
    o = subprocess.run(['fincore', '--bytes', '--noheadings', '--output', 'RES', f],
                       capture_output=True, text=True).stdout.split()
    return int(o[0]) if o else -1


def evict(f):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


def heavy_columns(scatter, k):
    t = collections.Counter(); qs = collections.defaultdict(set)
    for l in open(scatter):
        r = json.loads(l)
        for c in r.get('calls', []):
            d, m, col, enc = c[0].split('|')
            if d == '0' and m in ('_raw_codes', '_raw_codes_range'):
                t[col] += c[3]; qs[col].add(r['q'])
    return [(col, ms, sorted(qs[col])) for col, ms in t.most_common(k)]


if __name__ == '__main__':
    db, scatter, trace_bin = sys.argv[1:4]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    path = glob.glob(os.path.join(db, '*.wdb'))[0]
    cols = heavy_columns(scatter, k)
    print('%-18s %-6s %11s %10s | %8s %8s %8s | %7s %7s %6s %6s %6s' % (
        'column', 'enc', 'board ms', 'V', 'cold ms', 'warm ms', 'read MB', 'stored', 'log2 V', 'H0', 'runs', 'GB/s'))
    frames = []
    for col, board_ms, qs in cols:
        evict(path); r0 = resident(path)
        s = wdb_engine.Segment(path); c = s.cols[col]
        t = time.perf_counter(); cc = np.asarray(s._raw_codes(col)); cold = (time.perf_counter() - t) * 1e3
        rb = resident(path) - r0
        s2 = wdb_engine.Segment(path)
        t = time.perf_counter(); s2._raw_codes(col); warm = (time.perf_counter() - t) * 1e3
        N = cc.size; V = int(c['V'])
        cnt = np.bincount(cc.astype(np.int64), minlength=V); p = cnt[cnt > 0] / N
        H0 = float(-(p * np.log2(p)).sum())
        runs = float((cc[1:] == cc[:-1]).mean())
        enc = c.get('code_enc', 0)
        print('%-18s %-6s %11.0f %10d | %8.0f %8.0f %8.1f | %7.2f %7.2f %6.2f %5.0f%% %6.2f   queries %s' % (
            col, '%s/m%s' % (enc, c.get('mode')), board_ms, V, cold, warm, rb / 1e6, rb * 8 / N,
            np.log2(max(V, 2)), H0, 100 * runs, (rb / 1e9) / max(1e-9, (cold - warm) / 1e3), qs), flush=True)
        if enc == 3 and 'boffs' in c:
            frames.append((col, c))
        del cc, s, s2
    s = wdb_engine.Segment(path)
    for col, c in frames[:5]:
        c = s.cols[col]; bo = c['boffs']; cs = c['cstart']; nb = bo.size - 1
        for j in sorted({100 % nb, nb // 2}):
            fr = bytes(s.buf[cs + int(bo[j]):cs + int(bo[j + 1])])
            ZA.report('%s codes frame %d/%d (cwidth %d)' % (col, j, nb, c['cwidth']), fr, int(c['cwidth']), trace_bin)
