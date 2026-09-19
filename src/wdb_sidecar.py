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

THE SWITCH (Jackson's operator law): sidecars are an EXTENSION the operator turns on, not a
tax the engine levies in secret. A database's catalog carries 'sidecars': 'on' | 'off'; off
means no derived file is ever born (the engine answers from segments + the RAM shelf), so the
number on disk is the number that was loaded. WDB_SIDECARS=0|1 in the environment overrides
the catalog (an A/B without touching the realm). A catalog without the key is 'on' -- every
realm born before the switch keeps its roads. `wdb sidecars DB on|off|status|drop`.
THE SENTINEL: while off, Database.run compares the directory before and after each query;
a newborn is a missed gate -- removed and named on stderr, raised under WDB_SIDECAR_STRICT=1
(the suite runs strict: a silent gate is a defect, not overhead).
"""
import os, re, sys, json, time

# suffix regex -> (family, scale, module that births it)
FAMILIES = [
    (r'\.jptr\.npy$',            'road',           'N',  'wdb_join._hash_pointer'),
    (r'\.jptr\.npy\.mark$',      'road-mark',      '-',  'wdb_join._hash_pointer'),
    (r'\.jptr\.npy\.no$',        'road-refusal',   '-',  'wdb_join._hash_pointer (not a pointer: remembered)'),
    (r'\.cnt\.npy$',             'census',         'V',  'wdb_join (count sidecar)'),
    (r'\.txz\.bin$',             'text-buffer',    'V',  'wdb_wherescan (joined dictionary text, zstd frames)'),
    (r'\.txi\.npy$',             'text-frames',    'V',  'wdb_wherescan (frame index)'),
    (r'\.clen\.npy$',            'length',         'V',  'wdb_wherescan (character lengths)'),
    (r'\.rg-[0-9a-f]{8}\.npz$',  'regex-group',    'V',  'wdb_regexgroup (the road\'s result)'),
    (r'\.srank\.npy$',           'string-rank',    'N',  'wdb_semijoin (rank of each row in sorted order)'),
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
    (r'\.fkptr\.[A-Za-z0-9_]+$', 'fk-pointer',     'N',  'wdb_fkptr'),
    (r'\.lmap\.[A-Za-z0-9_]+$',  'lower-map',      'V',  'wdb_lmap'),
    (r'\.nline\.[A-Za-z0-9_]+$', 'number-line',    'V',  'wdb_nline'),
    (r'^tier2__.*\.bin$',        'tier-shelf',     'N',  'wdb_pairdistinct'),
    (r'\.npz$',                  'npz-shelf',      '?',  'various'),
]
DATA_FILES = ('catalog.json', 'sidecars.json', 'shelves.json')      # never sidecars: the realm itself and its ledgers
_DATA_SUFFIXES = ('.wdb', '.jsonl', '.presence', '.overrides', '.parquet',   # segments, journals, presence, DML hot buffers
                  '.cluster', '.cube',                                       # written by the ENCODER: part of the load, not derived
                  '.tmp', '.partial')                                        # a birth in flight (the rename law)


def is_data_file(name):
    """the files that ARE the database (loaded by the encoder or written by DML) -- everything else in the
    directory is derived from them and may be dropped"""
    return name in DATA_FILES or name.endswith(_DATA_SUFFIXES) or '.tmp.' in name
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


_SETTING_CACHE = {}


def _env_setting():
    env = os.environ.get('WDB_SIDECARS')
    if env is None or env.strip() == '':
        return None
    return 'off' if env.strip().lower() in ('0', 'off', 'no', 'false') else 'on'


def setting(dbdir):
    """'on' | 'off' for this database: WDB_SIDECARS in the environment wins; else the catalog's
    'sidecars' key; a catalog without the key is 'on'. Cached under the catalog stamp."""
    env = _env_setting()
    if env is not None:
        return env
    stamp = _catalog_stamp(dbdir)
    hit = _SETTING_CACHE.get(dbdir)
    if hit is not None and stamp is not None and hit[0] == stamp:
        return hit[1]
    v = 'on'
    try:
        with open(os.path.join(dbdir, 'catalog.json')) as f:
            v = 'off' if str(json.load(f).get('sidecars', 'on')).lower() == 'off' else 'on'
    except Exception:
        pass
    if stamp is not None:
        _SETTING_CACHE[dbdir] = (stamp, v)
    return v


def births_on(dbdir):
    """THE SWITCH, asked at every birth site: False means write nothing derived to disk."""
    return setting(dbdir) == 'on'


def set_setting(dbdir, value):
    """write 'sidecars': on|off into the catalog (the operator's decision, persisted with the realm)"""
    value = 'off' if str(value).strip().lower() in ('0', 'off', 'no', 'false') else 'on'
    p = os.path.join(dbdir, 'catalog.json')
    with open(p) as f:
        data = json.load(f)
    data['sidecars'] = value
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    stamp_moved(dbdir); _SETTING_CACHE.pop(dbdir, None); _fresh_cache.clear(); _exists_cache.clear()
    return value


def drop(dbdir, dry=False, print_out=True):
    """delete EVERY derived file in the realm (all families, fresh or stale, plus the birth ledgers)
    and turn the switch off: the directory is the loaded database again."""
    try:
        names = os.listdir(dbdir)
    except FileNotFoundError:
        return 0, 0
    gone = 0; freed = 0
    for n in sorted(names):
        if is_data_file(n) and n != 'shelves.json' and n != 'sidecars.json':
            continue
        p = os.path.join(dbdir, n)
        if not os.path.isfile(p):
            continue
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if print_out:
            print('  %s %s (%.1f MB)' % ('would remove' if dry else 'removing', n, sz / 1e6))
        if not dry:
            try:
                os.remove(p); gone += 1; freed += sz
            except OSError:
                pass
    if not dry:
        set_setting(dbdir, 'off')
    if print_out:
        print('DROP %s: removed %d files, freed %.2f GB; sidecars %s' % (dbdir, gone, freed / 1e9, 'off' if not dry else '(dry run)'))
    return gone, freed


_SENTINEL = {}


class SidecarBornWhileOff(Exception):
    pass


def sentinel_before(dbdir):
    """THE SENTINEL, armed: remember the directory when the switch is off (nothing when on)"""
    if births_on(dbdir):
        return
    try:
        _SENTINEL[dbdir] = frozenset(os.listdir(dbdir))
    except OSError:
        _SENTINEL.pop(dbdir, None)


def sentinel_after(dbdir):
    """THE SENTINEL, read: a derived file that appeared while the switch was off is a missed gate.
    It is removed (the operator's number on disk holds) and named; the suite runs strict and raises."""
    before = _SENTINEL.pop(dbdir, None)
    if before is None:
        return
    try:
        now = os.listdir(dbdir)
    except OSError:
        return
    born = [n for n in now if n not in before and not is_data_file(n)]
    if not born:
        return
    for n in born:
        try:
            os.remove(os.path.join(dbdir, n))
        except OSError:
            pass
    msg = 'SIDECAR BORN WHILE OFF (a missed gate): %s' % ', '.join(sorted(born))
    if os.environ.get('WDB_SIDECAR_STRICT'):
        raise SidecarBornWhileOff(msg)
    print('wdb: ' + msg, file=sys.stderr, flush=True)


def may_birth(dbdir, nbytes, what=''):
    """THE DISK GATE: a birth that would push the db's sidecars past the budget is
    refused by name (the engine still answers -- without the sidecar). THE SWITCH first."""
    if not births_on(dbdir):
        raise BirthRefused('sidecars are off for %s: %s not born (wdb sidecars %s on)' % (dbdir, what or 'a birth', dbdir))
    _by, total = stats(dbdir, print_out=False)
    if total + nbytes > budget_bytes():
        raise BirthRefused('sidecar budget %.0f GB would be exceeded by %s (+%.2f GB on %.2f GB)' % (
            budget_bytes() / 1e9, what or 'a birth', nbytes / 1e9, total / 1e9))
    return True


class BirthRefused(Exception):
    pass


_fresh_cache = {}
_stamp_cache = {}
_exists_cache = {}


def exists(path, neg_ttl=2.0):
    """THE REMEMBERED EXISTENCE: sidecar loaders asked the filesystem the same question ~53 times
    per query. A positive answer holds until the directory's catalog stamp moves; a negative
    one for neg_ttl seconds (a birth can flip it)."""
    import time as _t
    d = os.path.dirname(path); stamp = _catalog_stamp(d)
    hit = _exists_cache.get(path)
    now = _t.monotonic()
    if hit is not None:
        ok, st, t = hit
        if ok and st == stamp: return True
        if not ok and now - t < neg_ttl: return False
    ok = os.path.exists(path)
    _exists_cache[path] = (ok, stamp, now)
    if len(_exists_cache) > 20000: _exists_cache.clear()
    return ok


def stamp_moved(dbdir):
    """an in-process writer moved the stamp: forget the cached one at once (the ttl below only covers
    writers in OTHER processes -- a DML within the ttl window was invisible to its own process)"""
    _stamp_cache.pop(dbdir, None)


def _catalog_stamp(dbdir, ttl=0.02):
    """the catalog's (mtime, size), re-read at most every ttl seconds -- 422 stat calls per eight
    queries were 0.13s of a 1.97s profile, each re-checking a stamp that had not moved"""
    import time as _t
    now = _t.monotonic(); hit = _stamp_cache.get(dbdir)
    if hit is not None and now - hit[0] < ttl: return hit[1]
    try:
        st = os.stat(os.path.join(dbdir, 'catalog.json')); stamp = (st.st_mtime_ns, st.st_size)
    except Exception:
        stamp = None
    _stamp_cache[dbdir] = (now, stamp)
    return stamp


def is_fresh(dbdir, fname):
    """A sidecar is TRUE only if it is newer than its segment (and its .mark, if any, matches).
    THE FRESHNESS CACHE: the verdict is remembered per process, keyed by the catalog's stamp --
    a JOB query made 51 stat calls (20ms of 160) re-checking sidecars that cannot have changed
    while the catalog has not."""
    stamp = _catalog_stamp(dbdir)
    ck = (dbdir, fname)
    hit = _fresh_cache.get(ck)
    if hit is not None and hit[0] == stamp and stamp is not None:
        return hit[1]
    v = _is_fresh_uncached(dbdir, fname)
    if stamp is not None: _fresh_cache[ck] = (stamp, v)
    return v


def _is_fresh_uncached(dbdir, fname):
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
