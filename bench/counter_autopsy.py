"""THE COUNTER AUTOPSY (Jackson's hunch, 2026-09-24): in CounterID's zstd frames, is there a
generative rule that places the code of 62 in its positions -- could a filter on the value be
answered by reading the rules instead of rebuilding every row?

For the frames holding the most rows of the target value: zstd's own decoder (traced, as in
bench/zstd_autopsy.py) gives every sequence (literal length, copy length, copy distance). Each
output byte gets its provenance; each 2-byte code gets a class:
  literal        -- both bytes written as literals (the value is in the file as itself)
  run            -- written by a copy of distance == code width (the previous value repeated)
  pattern        -- written by any other copy (bytes lifted from further back)
and for the target value's rows: how many distinct sequences wrote them, and how long their runs
are. Also: the frame's rows in runs of equal codes (the data's own shape, whatever the codec).

Usage: PYTHONPATH=src python bench/counter_autopsy.py DB_DIR TRACE_BIN COLUMN VALUE [nframes]
"""
import sys, os, glob, collections
import numpy as np
import wdb_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zstd_autopsy as ZA


def classify(n, events, w):
    """per output byte: 0 literal, 1 run copy (distance == w), 2 other copy; and the sequence index"""
    kind = np.full(n, -1, np.int8); seq = np.full(n, -1, np.int64)
    op = 0; si = 0
    for e in events:
        if e[0] == 'seq':
            _, ll, ml, off = e
            kind[op:op + ll] = 0; seq[op:op + ll] = si; op += ll
            kind[op:op + ml] = 1 if off == w else 2; seq[op:op + ml] = si; op += ml
            si += 1
        elif e[0] == 'last':
            kind[op:op + e[1]] = 0; seq[op:op + e[1]] = si; op += e[1]
    assert op == n, ('trace did not rebuild the frame', op, n)
    return kind, seq, si


if __name__ == '__main__':
    db, trace_bin, col, value = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    nfr = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    seg = wdb_engine.Segment(glob.glob(os.path.join(db, '*.wdb'))[0])
    c = seg.cols[col]
    dv = np.asarray(seg._typed_dict(col))
    code = int(np.searchsorted(dv, value)); assert dv[code] == value, ('value not in dictionary', value)
    w = int(c['cwidth']); BR = int(c['BR']); bo = c['boffs']; cs = int(c['cstart'])
    codes = np.asarray(seg._raw_codes(col))
    hit = codes == code
    per = np.add.reduceat(hit.astype(np.int64), np.arange(0, codes.size, BR))
    print('%s: enc %s, code width %d bytes, %d rows per frame, %d frames; value %d = code %d, %d rows (%.2f%%), in %d frames'
          % (col, c.get('code_enc'), w, BR, per.size, value, code, int(hit.sum()), 100 * hit.mean(), int((per > 0).sum())))
    tot = collections.Counter()
    for j in np.argsort(-per)[:nfr]:
        j = int(j)
        fr = bytes(seg.buf[cs + int(bo[j]):cs + int(bo[j + 1])])
        blocks = ZA.parse_frame(fr)
        data, events = ZA.trace(trace_bin, fr)
        n = len(data); nv = n // w
        vals = np.frombuffer(data, dtype={1: np.uint8, 2: np.uint16, 4: np.uint32}[w])
        assert np.array_equal(vals, codes[j * BR:j * BR + nv]), 'frame decode differs from the column'
        kind, seq, nseq = classify(n, events, w)
        kv = kind.reshape(nv, w); sv = seq.reshape(nv, w)
        cls = np.where((kv == 0).all(1), 'literal', np.where((kv == 1).all(1), 'run', 'pattern'))
        mine = vals == code
        runs_eq = int((vals[1:] != vals[:-1]).sum()) + 1
        # runs of the target value in the frame
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mine.astype(np.int8), [0]])))
        rl = edges[1::2] - edges[::2]
        tseq = np.unique(sv[mine].ravel())
        lit_kind = blocks[0].get('lit_type')
        print('\n frame %d: %d bytes -> %d rows; %d sequences (copy rules); literals %s; the column changes value %d times (%d runs, mean run %.1f rows)'
              % (j, len(fr), nv, nseq, lit_kind, runs_eq - 1, runs_eq, nv / runs_eq))
        print('   ALL rows by provenance:   ' + '  '.join('%s %.1f%%' % (k, 100 * (cls == k).mean()) for k in ('literal', 'run', 'pattern')))
        print('   value %d: %d rows (%.1f%% of the frame) in %d runs (mean %.1f rows, longest %d); written by %d distinct sequences'
              % (value, int(mine.sum()), 100 * mine.mean(), rl.size, rl.mean() if rl.size else 0, rl.max() if rl.size else 0, tseq.size))
        print('   value %d rows by provenance: ' % value + '  '.join('%s %.1f%%' % (k, 100 * (cls[mine] == k).mean()) for k in ('literal', 'run', 'pattern')))
        # does a run of the target start with a literal (the value in place) and continue by run copies?
        starts = edges[::2]
        st_cls = collections.Counter(cls[starts].tolist())
        print('   first row of each run of %d: %s' % (value, dict(st_cls)))
        for k in ('literal', 'run', 'pattern'):
            tot[k] += int((cls[mine] == k).sum())
        tot['rows'] += int(mine.sum()); tot['runs'] += int(rl.size); tot['seqs'] += int(tseq.size)
    print('\nOVER THESE FRAMES: value %d rows %d in %d runs, written by %d sequences; literal %d, run-copy %d, pattern-copy %d'
          % (value, tot['rows'], tot['runs'], tot['seqs'], tot['literal'], tot['run'], tot['pattern']))
