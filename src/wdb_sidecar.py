"""THE SIDECAR REGISTRY: one place that knows every derived artifact on disk --
what family it belongs to, which segment and column it derives from, how big it
is, and whether it is still TRUE (a sidecar older than its segment is stale by
construction; families with a .mark carry their own birthmark too).

    python3 -m wdb_sidecar stats  DBDIR        # by family: count, bytes, stale, orphans
    python3 -m wdb_sidecar vacuum DBDIR        # delete stale + orphaned sidecars
    python3 -m wdb_sidecar manifest DBDIR      # write DBDIR/sidecars.json
    python3 -m wdb_sidecar audit DBDIR         # storage modes vs cardinality: misfits named

Births go through may_birth(): a disk budget (WDB_SIDECAR_GB, default 64) refuses
new births with a NAMED decline instead of filling the disk.
"""
import os, re, sys, json, time

# suffix regex -> (family, scale, module that births it)
FAMILIES = [
    (r'\.jptr\.npy$',            'road',           'N',  'wdb_join._hash_pointer'),
    (r'\.jptr\.npy\.mark$',      'road-mark',      '-',  'wdb_join._hash_pointer'),
    (r'\.cnt\.npy$',             'census',         'V',  'wdb_join (count sidecar)'),
    (r'\.txz\.bin$',             'text-buffer',    'V',  'wdb_wherescan (joined dictionary text, zstd frames)'),
    (r'\.txi\.npy$',             'text-frames',    'V',  'wdb_wherescan (frame index)'),
    (r'\.txo\.npy$',             'text-offsets',   'V',  'wdb_wherescan (joined dictionary offsets)'),
    (r'\.ptrep$',                'pointer-rep',    'N',  'wdb_ptrep'),
    (r'\.pt2$',                  'pair-table',     'V2', 'wdb_pairfold / diskpair'),
    (r'\.plist$',                'posting-list',   'N',  'wdb_plist'),
    (r'\.gbc[0-9]*$',            'groupby-shelf',  'V',  'wdb_gbshelf'),
    (r'\.gbp$',                  'groupby-pair',   'V2', 'wdb_gbshelf'),
    (r'\.gd[cp]$',               'group-distinct', 'V',  'wdb_gdsidecar'),
    (r'\.gd-.*\.npz$',           'group-distinct', 'V',  'wdb_gdsidecar'),
    (r'\.bst\.npz$',             'bitset',         'N',  'wdb_bsi'),
    (r'\.fpm$',                  'fpm',            'V',  'wdb_fpm'),
    (r'\.cluster$',              'cluster',        'V',  'wdb_cluster'),
    (r'\.inv\.(u|offs|order|rank)\.npy$', 'reverse-road', 'N', 'wdb_semijoin.inverted'),
    (r'\.npz$',                  'npz-shelf',      '?',  'various'),
]
_SEG = re.compile(r'^(?P<seg>[A-Za-z0-9_]+_[0-9]+\.wdb)\.(?P<rest>.+)$')
_UNION = re.compile(r'^(?P<table>[A-Za-z0-9_]+)\.union-(?P<hash>[0-9a-f]{12})\.wdb\.(?P<rest>.+)$')


def current_union_hashes(dbdir):
    """THE UNION BIRTHMARK, read back: for every multi-segment table in the catalog, the hash
    a union of its CURRENT segment set would carry. A union sidecar whose hash is not in this
    set was born under a segment set that no longer exists -- stale by construction."""
    import json, hashlib
    out = {}
    try:
        cat = json.load(open(os.path.join(dbdir, 'catalog.json')))
    except Exception:
        return out
    for t, info in cat.get('tables', {}).items():
        segs = info.get('segments') or []
        if len(segs) < 2: continue
        paths = [os.path.join(dbdir, sf) for sf in segs]
        try:
            stamp = hashlib.sha1('|'.join('%s:%d:%d' % (p, int(os.stat(p).st_mtime_ns), os.stat(p).st_size) for p in paths).encode()).hexdigest()[:12]
        except FileNotFoundError:
            continue
        out[t] = stamp
    return out


def classify(fname):
    for rx, fam, scale, mod in FAMILIES:
        if re.search(rx, fname):
            return fam, scale, mod
    return None


_union_hashes_cache = {}


def scan(dbdir):
    _union_hashes_cache.pop(dbdir, None)
    """Every sidecar in dbdir with its family, source segment, size, and truth."""
    out = []
    try:
        names = os.listdir(dbdir)
    except FileNotFoundError:
        return out
    segs = {n for n in names if n.endswith('.wdb')}
    for n in names:
        if n.endswith('.wdb') or n in ('catalog.json', 'sidecars.json') or n.endswith('.jsonl'):
            continue
        cl = classify(n)
        if cl is None:
            continue
        fam, scale, mod = cl
        p = os.path.join(dbdir, n)
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        m = _SEG.match(n)
        seg = m.group('seg') if m else None
        col = None
        if m:
            rest = m.group('rest')
            col = rest.split('.')[0].split('__')[0]
        status = 'ok'
        mu = _UNION.match(n) if m is None else None
        if mu is not None:
            # a UNION sidecar: its segment is virtual; fresh iff its hash is the table's current one
            seg = '%s.union-%s.wdb' % (mu.group('table'), mu.group('hash')); col = mu.group('rest').split('.')[0].split('__')[0]
            cur = _union_hashes_cache.get(dbdir)
            if cur is None: cur = _union_hashes_cache[dbdir] = current_union_hashes(dbdir)
            status = 'ok' if cur.get(mu.group('table')) == mu.group('hash') else 'stale'
        elif seg is None or seg not in segs:
            status = 'orphan'
        else:
            try:
                if os.stat(os.path.join(dbdir, seg)).st_mtime > st.st_mtime + 1:
                    status = 'stale'                       # older than its source: false by construction
            except FileNotFoundError:
                status = 'orphan'
        out.append({'file': n, 'family': fam, 'scale': scale, 'born_by': mod, 'segment': seg, 'column': col,
                    'bytes': st.st_size, 'mtime': st.st_mtime, 'status': status})
    return out


def stats(dbdir, print_out=True):
    rows = scan(dbdir)
    by = {}
    for r in rows:
        d = by.setdefault(r['family'], {'count': 0, 'bytes': 0, 'stale': 0, 'orphan': 0, 'scale': r['scale'], 'born_by': r['born_by']})
        d['count'] += 1; d['bytes'] += r['bytes']
        if r['status'] == 'stale': d['stale'] += 1
        if r['status'] == 'orphan': d['orphan'] += 1
    total = sum(r['bytes'] for r in rows)
    if print_out:
        print('SIDECARS in %s: %d files, %.2f GB' % (dbdir, len(rows), total / 1e9))
        for fam, d in sorted(by.items(), key=lambda kv: -kv[1]['bytes']):
            print('  %-16s x%-4d %7.2f GB  scale=%-2s stale=%d orphan=%d  (%s)' % (
                fam, d['count'], d['bytes'] / 1e9, d['scale'], d['stale'], d['orphan'], d['born_by']))
    return by, total


def manifest(dbdir):
    rows = scan(dbdir)
    doc = {'written': time.time(), 'dbdir': dbdir, 'count': len(rows), 'bytes': sum(r['bytes'] for r in rows), 'sidecars': rows}
    with open(os.path.join(dbdir, 'sidecars.json'), 'w') as f:
        json.dump(doc, f)
    return doc


def vacuum(dbdir, dry=False):
    """Delete stale (older than their segment) and orphaned (segment gone) sidecars."""
    rows = scan(dbdir)
    gone = 0; freed = 0
    for r in rows:
        if r['status'] in ('stale', 'orphan'):
            p = os.path.join(dbdir, r['file'])
            print('  %s %-8s %s (%.1f MB)' % ('would remove' if dry else 'removing', r['status'], r['file'], r['bytes'] / 1e6))
            if not dry:
                try:
                    os.remove(p); gone += 1; freed += r['bytes']
                except OSError:
                    pass
    print('VACUUM %s: removed %d, freed %.2f GB' % (dbdir, gone, freed / 1e9))
    return gone, freed


def audit(dbdir, print_out=True):
    """THE MODE AUDIT: every column's storage mode against its cardinality -- a sequence
    (mode 4) on a narrow column, inline strings (mode 5) on a low-cardinality column, a
    dictionary on a unique id -- named with a re-encode recommendation. Stale encodes
    (before a guard was added) show up here instead of as slow queries."""
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wdb_engine import Segment
    import numpy as np
    findings = []
    for n in sorted(os.listdir(dbdir)):
        if not n.endswith('.wdb'): continue
        try:
            seg = Segment(os.path.join(dbdir, n))
        except Exception as e:
            findings.append((n, '*', 'unreadable: %s' % str(e)[:60])); continue
        N = int(seg.N)
        for col, c in seg.cols.items():
            mode = c.get('mode'); V = int(c.get('V') or 0); dt = c.get('dt')
            if mode == 4:
                try:
                    vals = np.asarray(seg._seq_decode(c))
                    span = int(vals.max() - vals.min()) if vals.size else 0
                    if span < (1 << 16):
                        findings.append((n, col, 'MODE 4 (sequence) on a narrow column (span %d): a dictionary column; every predicate decodes %s rows -- RE-ENCODE' % (span, format(N, ','))))
                except Exception:
                    pass
            elif mode == 5 and dt == 1 and N > 100_000:
                findings.append((n, col, 'mode 5 (inline strings), %s rows: LIKE/= walk the text (cached under the shelf); fine if near-unique' % format(N, ',')))
            elif mode in (0, 1, 2) and dt == 0 and V == N and N > 1_000_000:
                findings.append((n, col, 'dictionary on a unique integer (V == N = %s): a sequence encode would be smaller' % format(N, ',')))
    if print_out:
        print('MODE AUDIT %s: %d findings' % (dbdir, len(findings)))
        for n, col, msg in findings:
            print('  %-24s %-18s %s' % (n, col, msg))
    return findings


def budget_bytes():
    try:
        return int(float(os.environ.get('WDB_SIDECAR_GB', '64')) * 1e9)
    except Exception:
        return 64 * 10**9


def may_birth(dbdir, nbytes, what=''):
    """THE DISK GATE: a birth that would push the db's sidecars past the budget is
    refused by name (the engine still answers -- without the sidecar)."""
    _by, total = stats(dbdir, print_out=False)
    if total + nbytes > budget_bytes():
        raise BirthRefused('sidecar budget %.0f GB would be exceeded by %s (+%.2f GB on %.2f GB)' % (
            budget_bytes() / 1e9, what or 'a birth', nbytes / 1e9, total / 1e9))
    return True


class BirthRefused(Exception):
    pass


def is_fresh(dbdir, fname):
    """A sidecar is TRUE only if it is newer than its segment (and its .mark, if any, matches)."""
    p = os.path.join(dbdir, fname)
    mu = _UNION.match(fname)
    if mu is not None:
        return os.path.exists(p) and current_union_hashes(dbdir).get(mu.group('table')) == mu.group('hash')
    m = _SEG.match(fname)
    if not m or not os.path.exists(p):
        return False
    seg = os.path.join(dbdir, m.group('seg'))
    try:
        return os.stat(seg).st_mtime <= os.stat(p).st_mtime + 1
    except FileNotFoundError:
        return False


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == 'stats': stats(d)
    elif cmd == 'audit': audit(d)
    elif cmd == 'vacuum': vacuum(d, dry='--dry' in sys.argv)
    elif cmd == 'manifest': print(json.dumps({k: v for k, v in manifest(d).items() if k != 'sidecars'}))
    else: print(__doc__)
