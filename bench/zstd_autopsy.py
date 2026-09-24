"""THE ZSTD AUTOPSY (Jackson, 2026-09-24): watch zstd rebuild real frames of the database, rule by
rule, and ask of every stored value: is it already in place (every byte a literal), one rule away
(a copy of literal bytes), or at the end of a chain of copies -- and how much of the frame's work
has to run before it can be plucked.

zstd's own decoder (built with its internal trace: DEBUGLEVEL=6, bench dir /workspace/zaut/trace)
logs every sequence it executes -- literal length, match length, match distance (repeat codes
already resolved). The frame and block headers and each literals-section header are parsed here
from the frame bytes (RFC 8878). From the sequences every output byte gets a provenance: a literal
(depth 0), or a copy of an earlier byte (depth = its source's depth + 1).

Usage: PYTHONPATH=src python bench/zstd_autopsy.py DB_DIR TRACE_BIN
"""
import sys, os, glob, re, subprocess, tempfile, collections
import numpy as np
import wdb_engine

LIT_TYPES = {0: 'raw', 1: 'rle', 2: 'huffman', 3: 'huffman-reused-table'}
BLK_TYPES = {0: 'raw', 1: 'rle', 2: 'compressed', 3: 'reserved'}


def parse_frame(b):
    """frame header, then every block: (type, size, literals type, literal count, streams)"""
    assert b[:4] == b'\x28\xb5\x2f\xfd', 'not a zstd frame'
    fhd = b[4]; p = 5
    fcs_flag = fhd >> 6; single = (fhd >> 5) & 1; did = fhd & 3
    if not single: p += 1
    p += [0, 1, 2, 4][did]
    p += [1 if single else 0, 2, 4, 8][fcs_flag]
    blocks = []
    while True:
        h = b[p] | (b[p + 1] << 8) | (b[p + 2] << 16); p += 3
        last = h & 1; bt = (h >> 1) & 3; bs = h >> 3
        info = {'type': BLK_TYPES[bt], 'csize': bs if bt != 1 else 1}
        if bt == 2:
            l0 = b[p]; lt = l0 & 3; sf = (l0 >> 2) & 3
            info['lit_type'] = LIT_TYPES[lt]
            if lt in (0, 1):
                if sf in (0, 2): n = l0 >> 3
                elif sf == 1: n = (l0 >> 4) | (b[p + 1] << 4)
                else: n = (l0 >> 4) | (b[p + 1] << 4) | (b[p + 2] << 12)
                info['lit_n'] = n; info['lit_c'] = n if lt == 0 else 1; info['streams'] = 1
            else:
                x = int.from_bytes(bytes(b[p:p + 5]), 'little')
                if sf in (0, 1):
                    n = (x >> 4) & 0x3FF; c = (x >> 14) & 0x3FF; streams = 1 if sf == 0 else 4
                elif sf == 2:
                    n = (x >> 4) & 0x3FFF; c = (x >> 18) & 0x3FFF; streams = 4
                else:
                    n = (x >> 4) & 0x3FFFF; c = (x >> 22) & 0x3FFFF; streams = 4
                info['lit_n'] = n; info['lit_c'] = c; info['streams'] = streams
        blocks.append(info)
        p += bs if bt != 1 else 1
        if last: break
    return blocks


def trace(trace_bin, frame):
    with tempfile.NamedTemporaryFile(suffix='.zst', delete=False) as f:
        f.write(frame); fn = f.name
    out = fn + '.out'
    r = subprocess.run([trace_bin, fn, out], capture_output=True, text=True)
    data = open(out, 'rb').read(); os.unlink(fn); os.unlink(out)
    events = []   # ('seq', litL, matchL, off) | ('last', n) | ('block',)
    for line in r.stderr.splitlines():
        m = re.search(r'seq: litL=(\d+), matchL=(\d+), offset=(\d+)', line)
        if m: events.append(('seq', int(m.group(1)), int(m.group(2)), int(m.group(3)))); continue
        m = re.search(r'copy last literals[^:]*: (\d+)', line)
        if m: events.append(('last', int(m.group(1)))); continue
        if 'decoded block of size' in line: events.append(('block',))
    return data, events


def provenance(n, events, blocks):
    """depth[i] = 0 if output byte i is a literal, else 1 + depth of the byte it copies;
    seq_at[i] = how many sequences had been decoded when byte i was written;
    lit_at[i] = how many literal bytes had been decoded when byte i was written."""
    depth = np.full(n, -1, np.int32); seq_at = np.zeros(n, np.int64); lit_at = np.zeros(n, np.int64)
    op = 0; nseq = 0; nlit = 0; bi = 0
    raw_rle = [b for b in blocks if b['type'] != 'compressed']
    for e in events:
        if e[0] == 'seq':
            _, ll, ml, off = e; nseq += 1
            depth[op:op + ll] = 0; seq_at[op:op + ll] = nseq; lit_at[op:op + ll] = nlit + np.arange(1, ll + 1)
            op += ll; nlit += ll
            for k in range(ml):                          # byte by byte: a copy may overlap itself
                depth[op] = depth[op - off] + 1; seq_at[op] = nseq; lit_at[op] = nlit; op += 1
        elif e[0] == 'last':
            ll = e[1]
            depth[op:op + ll] = 0; seq_at[op:op + ll] = nseq; lit_at[op:op + ll] = nlit + np.arange(1, ll + 1)
            op += ll; nlit += ll
    assert not raw_rle or op == 0 or op == n, ('raw/rle blocks mixed with traced blocks: extend the walk', op, n)
    if op == 0 and raw_rle:                              # a raw block: every byte stored as itself
        depth[:] = 0; lit_at[:] = np.arange(1, n + 1)
    assert op == n or (op == 0 and raw_rle), ('the trace did not rebuild the frame', op, n)
    return depth, seq_at, lit_at, nseq, nlit


def report(name, frame, w, trace_bin):
    blocks = parse_frame(frame)
    data, events = trace(trace_bin, frame)
    n = len(data)
    depth, seq_at, lit_at, nseq, nlit = provenance(n, events, blocks)
    nv = n // w
    dv = depth[:nv * w].reshape(nv, w)
    maxd = dv.max(1); allit = (dv == 0).all(1)
    ends = np.arange(1, nv + 1) * w - 1                   # a value is ready when its last byte is written
    seq_frac = seq_at[ends] / max(1, nseq); lit_frac = lit_at[ends] / max(1, nlit)
    print('\n== %s: %d bytes -> %d bytes (%d values of %d bytes), %d blocks' % (name, len(frame), n, nv, w, len(blocks)))
    for b in blocks[:4]:
        print('   block %-10s csize %6d  literals %-20s %6s bytes -> %6s  streams %s' % (
            b['type'], b['csize'], b.get('lit_type', '-'), b.get('lit_n', '-'), b.get('lit_c', '-'), b.get('streams', '-')))
    if len(blocks) > 4: print('   ... %d more blocks' % (len(blocks) - 4))
    print('   sequences (copy rules): %d   literal bytes: %d (%.1f%% of output)   copied bytes: %d' % (
        nseq, nlit, 100 * nlit / n, n - nlit))
    ml = [e[2] for e in events if e[0] == 'seq']; off = [e[3] for e in events if e[0] == 'seq']
    if ml:
        print('   copy length: median %d  max %d   copy distance: median %d  max %d' % (
            int(np.median(ml)), max(ml), int(np.median(off)), max(off)))
    hist = collections.Counter(int(x) for x in depth)
    print('   byte depth: ' + '  '.join('%d: %.1f%%' % (d, 100 * c / n) for d, c in sorted(hist.items())[:8]))
    print('   VALUES: in place (all bytes literal) %.1f%%   one rule away %.1f%%   deeper %.1f%%   (max depth %d)' % (
        100 * allit.mean(), 100 * (maxd == 1).mean(), 100 * (maxd > 1).mean(), int(maxd.max())))
    pos = (dv == 0).mean(0)
    print('   share literal by byte position in the value (low byte first): ' + ' '.join('%.0f%%' % (100 * x) for x in pos))
    print('   work before a value is ready -- sequences run: median %.0f%%   literal bytes decoded: median %.0f%%' % (
        100 * np.median(seq_frac), 100 * np.median(lit_frac)))
    q = [1, 10, 50, 90]
    print('   pluck value k%% of the way in: ' + '  '.join('k=%d%%: %.0f%% of rules, %.0f%% of literals' % (
        k, 100 * seq_frac[min(nv - 1, nv * k // 100)], 100 * lit_frac[min(nv - 1, nv * k // 100)]) for k in q))


if __name__ == '__main__':
    db, trace_bin = sys.argv[1], sys.argv[2]
    seg = wdb_engine.Segment(glob.glob(os.path.join(db, '*.wdb'))[0])
    buf = seg.buf
    for nm in ('URLHash', 'UserID', 'WatchID', 'ClientIP', 'HID', 'EventTime'):
        c = seg.cols[nm]; zo = c['i2zoffs']; base = c['i2base']; nch = len(zo) - 1
        for j in sorted({min(10, nch - 1), nch // 2}):
            fr = bytes(buf[base + int(zo[j]):base + int(zo[j + 1])])
            report('%s dictionary chunk %d/%d' % (nm, j, nch), fr, 8, trace_bin)
    for nm in ('URL', 'Referer', 'UserID', 'RegionID'):
        c = seg.cols[nm]
        if c.get('code_enc', 0) != 3 or 'boffs' not in c:
            print('\n==', nm, 'codes are enc', c.get('code_enc'), '(not framed enc-3): skipped'); continue
        bo = c['boffs']; cs = c['cstart']
        for j in (100,):
            fr = bytes(buf[cs + int(bo[j]):cs + int(bo[j + 1])])
            report('%s codes frame %d (cwidth %d)' % (nm, j, c['cwidth']), fr, int(c['cwidth']), trace_bin)
