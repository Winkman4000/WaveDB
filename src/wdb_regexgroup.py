"""regexgroup: GROUP BY REGEXP_REPLACE(dict column) computed entirely at the dictionary level.

The Q28 shape: SELECT REGEXP_REPLACE(col, pat, repl) AS k, AVG(LENGTH(col)), COUNT(*),
MIN(col) ... WHERE col <> '' GROUP BY k HAVING COUNT(*) > N ORDER BY <alias> DESC LIMIT n.

Row data is never touched as strings: one parallel frame scan produces per-code row counts
(bincount lanes), and every string operation -- the regex, the lengths, the min -- happens once
per DISTINCT value against the dictionary, then aggregates by weight. MIN(col) is free: dicts
are value-sorted, so the first code carrying each label is its minimum. The regex applies to V
distinct values instead of N rows -- a ~10x reduction on ClickBench's Referer.
"""
import os
import numpy as np
import re
import zstandard as zstd
import sqlglot.expressions as E
from concurrent.futures import ThreadPoolExecutor
import wdb_sql
import wdb_policies as P

_FORK_BS = None
_FORK_RX = None
_FORK_REP = None


def _fork_chunk(se):
    rx = re.compile(_FORK_RX)
    return [rx.sub(_FORK_REP, v) for v in _FORK_BS[se[0]:se[1]]]

_ENABLED = True
_HITS = 0
_RUNROAD = 0
_SCAN_THREADS = 14


def enable():
    global _ENABLED; _ENABLED = True


def disable():
    global _ENABLED; _ENABLED = False


def _regex_key(p):
    """(col, pattern, repl) for REGEXP_REPLACE(col, 'pat', 'repl') [AS alias]."""
    inner = p.this if isinstance(p, E.Alias) else p
    if not isinstance(inner, E.RegexpReplace) or not isinstance(inner.this, E.Column):
        return None
    pat = inner.expression
    rep = inner.args.get('replacement')
    if not isinstance(pat, E.Literal) or not isinstance(rep, E.Literal):
        return None
    return inner.this.name, str(pat.this), str(rep.this)


def _avg_length(p):
    """colname for AVG(LENGTH(col))."""
    inner = p.this if isinstance(p, E.Alias) else p
    if not isinstance(inner, E.Avg):
        return None
    a = inner.this
    if (isinstance(a, E.Anonymous) and str(a.this).upper() == 'STRLEN'
            and a.expressions and isinstance(a.expressions[0], E.Column)):
        return a.expressions[0].name, 'STRLEN'
    if isinstance(a, E.Length) and isinstance(a.this, E.Column):
        return a.this.name
    return None


def detect(seg, tree, col_map):
    # enc-5 (patched buckets) columns: this path's direct block machinery predates the
    # species; decline so the _raw_codes route (exact, 8-lane) serves until v2 learns nibbles
    for _c5 in tree.find_all(E.Column):
        _n5 = (col_map or {}).get(_c5.name, _c5.name) if col_map else _c5.name
        if _n5 in seg.cols and seg.cols[_n5].get('code_enc') in (5, 6):
            return None
    if not _ENABLED:                    return None
    if not P.no_joins(tree):            return None
    if not P.no_select_distinct(tree):  return None
    if not P.no_deleted_rows(seg):      return None
    where = tree.args.get('where')
    if where is None:
        return None
    w = where.this
    if not isinstance(w, E.NEQ) or not isinstance(w.this, E.Column):
        return None
    wcol = col_map.get(w.this.name, w.this.name) if col_map else w.this.name
    wl = w.expression
    if not (isinstance(wl, E.Literal) and wl.this == ''):
        return None
    proj = tree.expressions
    lenfn = 'LENGTH'
    rk = None; cols = {}
    for pi, p in enumerate(proj):
        r = _regex_key(p)
        if r is not None:
            if rk is not None: return None
            col = col_map.get(r[0], r[0]) if col_map else r[0]
            rk = (pi, col, r[1], r[2]); continue
        al = _avg_length(p)
        if al is not None:
            fn9 = 'LENGTH'
            if isinstance(al, tuple):
                al, fn9 = al
            col = col_map.get(al, al) if col_map else al
            cols[pi] = ('AVG_LEN', col)
            lenfn = fn9; continue
        ak = wdb_sql._agg_kind(p)
        if ak is not None and ak[0] == 'COUNT_STAR':
            cols[pi] = ('COUNT_STAR', None); continue
        if ak is not None and ak[0] == 'MIN' and isinstance(ak[1], str):
            col = col_map.get(ak[1], ak[1]) if col_map else ak[1]
            cols[pi] = ('MIN_DICT', col); continue
        return None
    if rk is None:
        return None
    col = rk[1]
    if col != wcol:                      return None
    if not P.columns_exist(seg, col):    return None
    if seg._effective(col) is not None:  return None
    if seg.cols[col].get('mode') not in (0, 1):
        return None
    for pi, (kind, c2) in cols.items():
        if c2 is not None and c2 != col:
            return None                  # v1: every aggregate rides the regex column
    group = tree.args.get('group')
    if group is None or len(group.expressions) != 1:
        return None
    having = tree.args.get('having')
    hmin = None
    if having is not None:
        h = having.this
        if not isinstance(h, E.GT):
            return None
        hk = wdb_sql._agg_kind(h.this)
        if hk is None or hk[0] != 'COUNT_STAR':
            return None
        try:
            hmin = int(str(h.expression.this))
        except Exception:
            return None
    order = tree.args.get('order')
    osel = None
    if order is not None:
        if len(order.expressions) != 1 or not order.expressions[0].args.get('desc'):
            return None
        onm = order.expressions[0].this
        tgt = onm.name if isinstance(onm, E.Column) else None
        for pi2, p in enumerate(proj):
            if tgt is not None and wdb_sql._alias(p) == tgt:
                osel = pi2
        if osel is None:
            return None
    lim = wdb_sql._limit(tree)
    return {'col': col, 'pat': rk[2], 'rep': rk[3], 'rk_pi': rk[0], 'aggs': cols,
            'hmin': hmin, 'osel': osel, 'lim': lim, 'off': int(wdb_sql._offset(tree) or 0),
            'proj': proj, 'lenfn': lenfn}


def _code_counts(seg, col):
    """Row count per dict code: THE CENSUS (Segment.code_counts -- the .cnt.npy sidecar, born once)
    when the segment has it; otherwise one parallel pass over the blocked frames (5.6s on Referer
    every process: the registry already knew the answer)."""
    c = seg.cols[col]
    V = int(c['V'])
    try:
        cc9 = seg.code_counts(col)
        if cc9 is not None and len(cc9) >= V: return np.asarray(cc9)[:V]
    except Exception:
        pass
    if c.get('code_enc') != 3 or col in seg._codes:
        cc = np.asarray(seg._raw_codes(col))
        return np.bincount(cc, minlength=V)
    wdt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[c['cwidth']]
    bo = c['boffs']; base = c['cstart']; buf = seg.buf
    NB = bo.size - 1

    def work(js):
        dz = zstd.ZstdDecompressor()
        tab = np.zeros(V, np.int64)
        for j in js:
            raw = np.frombuffer(dz.decompress(buf[base + int(bo[j]):base + int(bo[j + 1])].tobytes()), dtype=wdt)
            tab += np.bincount(raw, minlength=V)
        return tab

    W = min(_SCAN_THREADS, NB) or 1
    with ThreadPoolExecutor(W) as ex:
        parts = list(ex.map(work, np.array_split(np.arange(NB), W)))
    tot = parts[0]
    for p in parts[1:]:
        tot += p
    return tot


def execute(seg, spec):
    global _HITS
    import pandas as pd
    col = spec['col']
    memo = seg.__dict__.setdefault('_rg_memo', {})
    mk = (col, spec['pat'], spec['rep'], spec.get('lenfn'))
    if mk in memo:
        counts, lens, lab_ids, uniq, empty_code = memo[mk]
    else:
        loaded = _load_sidecar(seg, col, spec)
        if loaded is not None:
            lens, lab_ids, uniq, empty_code = loaded
            counts = _code_counts(seg, col)
        else:
            counts, lens, lab_ids, uniq, empty_code = _derive(seg, col, spec)
            _save_sidecar(seg, col, spec, lens, lab_ids, uniq, empty_code)
        memo[mk] = (counts, lens, lab_ids, uniq, empty_code)
    return _emit(seg, spec, counts, lens, lab_ids, uniq, empty_code)


def _sidecar_path(seg, col, spec):
    import hashlib
    base = getattr(seg, 'path', None)
    if not base or str(os.path.basename(base)).find('.union-') >= 0: return None
    h = hashlib.sha1(('%s|%s|%s' % (spec['pat'], spec['rep'], spec.get('lenfn'))).encode()).hexdigest()[:8]
    return '%s.%s.rg-%s.npz' % (base, col, h)


def _load_sidecar(seg, col, spec):
    """THE REGEX-GROUP SIDECAR: the road's V-scale result (label id per value, the distinct
    labels, character lengths) born once and loaded in milliseconds -- 19.7M Referers took 15s
    to hostize on every fresh process (Q28 cold), for a result that never changes"""
    p = _sidecar_path(seg, col, spec)
    if not p or not os.path.exists(p): return None
    try:
        import wdb_sidecar
        if not wdb_sidecar.is_fresh(os.path.dirname(p), os.path.basename(p)): return None
        z = np.load(p, allow_pickle=False)
        lab_ids = z['lab_ids']; lens = z['lens']; ub = z['uniq_bytes'].tobytes(); uo = z['uniq_offs']
        uniq = np.array([ub[int(uo[i]):int(uo[i + 1])] for i in range(int(uo.size) - 1)], dtype=object)
        ec = int(z['empty_code'][0]); empty_code = None if ec < 0 else ec
        return lens, lab_ids, uniq, empty_code
    except Exception:
        return None


def _save_sidecar(seg, col, spec, lens, lab_ids, uniq, empty_code):
    p = _sidecar_path(seg, col, spec)
    if not p: return
    import wdb_sidecar
    if not wdb_sidecar.may_build(p):
        return                                   # THE VANILLA LAW, asked FIRST: the payload below
                                                 # turned 3.0M labels into bytes (3.4 s) to be refused
    try:
        ub = [v if isinstance(v, (bytes, bytearray)) else str(v).encode() for v in uniq]
        uo = np.zeros(len(ub) + 1, np.int64); np.cumsum([len(v) for v in ub], out=uo[1:])
        nbytes = int(np.asarray(lab_ids).nbytes + np.asarray(lens).nbytes + uo[-1] + uo.nbytes)
        if not wdb_sidecar.may_birth(os.path.dirname(p), nbytes, 'regex-group %s' % col): return
        np.savez(p + '.partial.npz', lab_ids=np.asarray(lab_ids, dtype=np.int32), lens=np.asarray(lens, dtype=np.int32),
                 uniq_bytes=np.frombuffer(b''.join(ub), dtype=np.uint8), uniq_offs=uo, empty_code=np.array([-1 if empty_code is None else int(empty_code)], dtype=np.int64))
        os.replace(p + '.partial.npz', p)                                    # the rename law
    except Exception:
        pass


_CANON_PAT = '^https?://(?:www\\.)?([^/]+)/.*$'


class _Labels:
    """the distinct labels as ONE byte blob: a label becomes bytes only when asked for (THE RETURN
    READ -- _emit asks only for the groups that survive HAVING)"""
    def __init__(self, blob, offs, rep):
        self.blob = blob; self.offs = offs; self.rep = rep
    def __len__(self):
        return int(self.rep.size)
    def __getitem__(self, g):
        r = int(self.rep[int(g)])
        return self.blob[int(self.offs[r]):int(self.offs[r + 1])].tobytes()


def _derive_runs_one_read(seg, col, spec):
    """THE THREE READS, Q28: two properties of every distinct Referer -- its length and its host --
    from ONE read of the dictionary as stored. Chunks decompress once, in parallel; the length
    walk and the host-run walk (the prefix-run road: a host is inherited while the shared prefix
    reaches past its slash) both run on the chunk while it is in cache; labels are grouped exactly
    in a kernel (hash buckets, byte-for-byte inside a bucket); only the surviving groups' labels
    ever become strings. The old road decompressed the dictionary twice, walked chunks serially and
    grouped 3.1M labels as Python objects (~7 s). Returns None (the caller falls back) on surprise."""
    import wdb_kernels as WK, wdb_strings, wdb_engine
    c = seg.cols.get(col)
    if c is None or c.get('R') is None or not c.get('chunked'):
        return None
    R = int(c['R']); nd = int(c.get('n_dict', c['V']))
    strlen = spec.get('lenfn') == 'STRLEN'
    lens = np.zeros(nd, np.int64)
    plan = wdb_strings._chunk_plan(seg, col)

    def _one(p):
        j, lo, n, _rl = p
        if c.get('fc3'):
            # THE THREE STREAMS: headers, mask and text inflate once each. Length from headers +
            # mask (or headers alone); the host from the headers -- a suffix is looked at only for a
            # newline while the run holds, and a string is rebuilt only where the host breaks.
            h = seg.fc_part(c, j, 'h'); hv = h.view(np.uint16); t = seg.fc_part(c, j, 't')
            assert hv.size == 2 * n, ('fc3 headers', col, j, hv.size, n)
            if strlen:
                lens[lo:lo + n] = hv[0::2].astype(np.int64) + hv[1::2]
            else:
                m = seg.fc_part(c, j, 'm')
                if int(WK.fc3_charlens(hv, m.view(np.uint64), np.int64(R), lens[lo:lo + n])) != n:
                    return None
            nl_free = not bool((t == 10).any())
            cap = n + 2
            lcap = t.size * 3 + (1 << 16)
            for _try in range(4):                # labels can outgrow the text: retry, tripled
                brk = np.zeros(cap, np.uint8); hend = np.full(cap, -1, np.int32)
                labuf = np.empty(lcap, np.uint8); laboff = np.empty(cap + 1, np.int64)
                meta = np.zeros(1, np.int64)
                nr = int(WK.fc3_hostruns(hv, t, np.int64(R), nl_free, brk, hend, labuf, laboff, meta))
                if int(meta[0]) != -1:
                    break
                lcap *= 3
            if int(meta[0]) != n:
                return None
            return brk[:n].copy(), labuf[:int(laboff[nr])].copy(), laboff[:nr + 1].copy(), nr
        # length: headers + mask (characters) or headers alone (bytes)
        m = (wdb_strings.bytelens_chunk if strlen else wdb_strings.charlens_chunk)(seg, col, p, lens[lo:lo + n])
        if int(m) != n:
            return None
        a = wdb_strings._chunk_bytes(seg, col, j)   # the interleaved layout: its own frames
        cap = a.size // 4 + 2
        lcap = a.size * 3 + (1 << 16)
        for _try in range(4):                    # labels can outgrow the fc bytes: retry, tripled
            brk = np.zeros(cap, np.uint8); hend = np.full(cap, -1, np.int32)
            labuf = np.empty(lcap, np.uint8); laboff = np.empty(cap + 1, np.int64)
            meta = np.zeros(1, np.int64)
            nr = int(WK.fc_hostruns(a, np.int64(R), brk, hend, labuf, laboff, meta))
            if int(meta[0]) != -1:
                break
            lcap *= 3
        if int(meta[0]) != n:
            return None
        return brk[:n].copy(), labuf[:int(laboff[nr])].copy(), laboff[:nr + 1].copy(), nr

    parts = list(wdb_engine._leaf_pool().map(_one, plan))
    if any(p is None for p in parts):
        return None
    nrs = np.array([p[3] for p in parts], np.int64)
    run_base = np.zeros(nrs.size + 1, np.int64); np.cumsum(nrs, out=run_base[1:])
    ent_run = np.concatenate([np.cumsum(p[0], dtype=np.int64) - 1 + run_base[k] for k, p in enumerate(parts)])
    if ent_run.size != nd:
        return None                              # layout surprise: fail closed
    blob = np.concatenate([p[1] for p in parts])
    lb = np.array([p[2][-1] for p in parts], np.int64)
    byte_base = np.zeros(lb.size + 1, np.int64); np.cumsum(lb, out=byte_base[1:])
    offs = np.concatenate([p[2][:-1] + byte_base[k] for k, p in enumerate(parts)] + [byte_base[-1:]])
    nrun = int(run_base[-1])
    h = np.empty(nrun, np.uint64); WK.label_hash(blob, offs, h)
    order = np.argsort(h)                        # bucket order only: groups are decided byte-exact
    gid = np.empty(nrun, np.int64); rep = np.empty(nrun, np.int64)
    G = int(WK.label_groups(blob, offs, h, order, gid, rep))
    lab_ids = gid[ent_run]
    counts = _code_counts(seg, col)
    empty_code = None
    e = np.nonzero(lens == 0)[0]
    if e.size:
        empty_code = int(e[0])
    return counts, lens, lab_ids, _Labels(blob, offs, rep[:G]), empty_code


def _derive_runs(seg, col, spec):
    """Jackson's prefix-run road for the canonical hostization: the sorted front-
    coded dict keeps one website's referers ADJACENT, so the label is constant
    while the copy-prefix reaches past the host's slash. Labels are produced by
    a byte walk (no string births, no regex); counts, lengths, MIN and HAVING
    all ride existing V-tables. Falls back (None) on any layout surprise."""
    import pandas as pd
    import wdb_kernels as WK
    c = seg.cols.get(col)
    if c is None or c.get('R') is None:
        return None
    lens = (seg.dict_bytelens(col) if spec.get('lenfn') == 'STRLEN'
            else seg.dict_charlens(col))
    if lens is None:
        return None
    R = int(c['R'])
    try:
        if c.get('chunked'):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as ex:
                raws = list(ex.map(lambda j: seg.fc_chunk(c, j, as_bytes=True), range(c['nch'])))
        else:
            raws = [seg._dz.decompress(c['z'])]
    except Exception:
        return None
    ent_lab = []
    all_labs = []
    for raw in raws:
        a = np.frombuffer(raw, dtype=np.uint8)
        cap = a.size // 4 + 2
        brk = np.zeros(cap, np.uint8)
        hend = np.full(cap, -1, np.int32)
        lcap = a.size * 3 + (1 << 16)
        nr = -1
        for _try in range(4):                    # labels can outgrow the fc bytes
            labuf = np.empty(lcap, np.uint8)     # (front-coding removed the very
            laboff = np.empty(cap + 1, np.int64)  # prefixes labels rebuild): retry
            meta = np.zeros(1, np.int64)         # with a tripled buffer on overflow
            nr = int(WK.fc_hostruns(a, np.int64(R), brk, hend, labuf, laboff, meta))
            if int(meta[0]) != -1:
                break
            lcap *= 3
        if int(meta[0]) == -1:
            return None
        nent = int(meta[0])
        rid = np.cumsum(brk[:nent], dtype=np.int64) - 1 + len(all_labs)
        ent_lab.append(rid)
        lb = labuf.tobytes()
        for r in range(nr):
            all_labs.append(lb[int(laboff[r]):int(laboff[r + 1])])
    ent_run = np.concatenate(ent_lab) if ent_lab else np.empty(0, np.int64)
    nd = int(c.get('n_dict', c['V']))
    if ent_run.size != nd:
        return None                              # layout surprise: fail closed
    run_lab, uniq = pd.factorize(np.array(all_labs, dtype=object), sort=False)
    lab_ids = run_lab[ent_run]
    counts = _code_counts(seg, col)
    lens = np.asarray(lens[:nd], np.int64)
    empty_code = None
    e = np.nonzero(lens == 0)[0]
    if e.size:
        empty_code = int(e[0])
    return counts, lens, lab_ids, uniq, empty_code


def _derive(seg, col, spec):
    import pandas as pd
    if spec.get('pat') == _CANON_PAT and spec.get('rep') in ('\\1', '\\g<1>'):
        r9 = _derive_runs_one_read(seg, col, spec)
        if r9 is None:
            r9 = _derive_runs(seg, col, spec)
        if r9 is not None:
            global _RUNROAD
            _RUNROAD += 1
            return r9
    counts = _code_counts(seg, col)
    vals = seg._typed_dict(col)
    # stay in BYTES end to end: no per-value decode (measured 15.7 s on 19.7M Referers);
    # compiled bytes regex; only the surviving group labels ever become str
    bs = [v if isinstance(v, (bytes, bytearray)) else str(v).encode() for v in vals]
    lens = np.fromiter((len(v) for v in bs), np.int64, len(bs))
    # duck's length() counts CHARACTERS: char len = byte len - UTF-8 continuation bytes,
    # counted in one vectorized pass over the joined buffer (no per-value decode)
    offs = np.zeros(len(bs) + 1, np.int64); np.cumsum(lens, out=offs[1:])
    cont = (np.frombuffer(b''.join(bs), dtype=np.uint8) & 0xC0) == 0x80
    cs = np.zeros(offs[-1] + 1, np.int64); np.cumsum(cont, out=cs[1:])
    if spec.get('lenfn') != 'STRLEN':
        lens = lens - (cs[offs[1:]] - cs[offs[:-1]])
    empty_code = None
    e = np.nonzero(lens == 0)[0]
    if e.size:
        empty_code = int(e[0])
    global _FORK_BS, _FORK_RX, _FORK_REP
    _FORK_BS, _FORK_RX, _FORK_REP = bs, spec['pat'].encode(), spec['rep'].encode()
    try:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(6) as pool:      # COW: children inherit bs, no copy in
            V2 = len(bs); step = (V2 + 5) // 6
            parts = pool.map(_fork_chunk, [(i, min(V2, i + step)) for i in range(0, V2, step)])
        labels = [x for part in parts for x in part]
    except Exception:
        rx = re.compile(_FORK_RX)
        labels = [rx.sub(_FORK_REP, v) for v in bs]
    finally:
        _FORK_BS = None
    lab_ids, uniq = pd.factorize(np.array(labels, dtype=object), sort=False)
    return counts, lens, lab_ids, uniq, empty_code


def _emit(seg, spec, counts, lens, lab_ids, uniq, empty_code):
    global _HITS
    col = spec['col']
    G = len(uniq)
    w = counts.astype(np.int64)
    if empty_code is not None:
        w = w.copy(); w[empty_code] = 0              # WHERE col <> ''
    gcnt = np.bincount(lab_ids, weights=w, minlength=G).astype(np.int64)
    glen = np.bincount(lab_ids, weights=w * lens, minlength=G)
    min_code = np.full(G, -1, np.int64)
    import wdb_kernels as _WKe
    _WKe.first_index(np.ascontiguousarray(lab_ids, dtype=np.int64), min_code)   # first code in code
                                                     # order = MIN value (order is differentiation)
    keep = np.nonzero(gcnt > (spec['hmin'] or 0))[0] if spec['hmin'] is not None else np.nonzero(gcnt > 0)[0]
    keep = keep[gcnt[keep] > 0]
    rows = []
    for g in keep:
        lab = uniq[g]
        lab = lab.decode('utf-8', 'replace') if isinstance(lab, (bytes, bytearray)) else str(lab)
        rows.append((lab, float(glen[g]) / gcnt[g], int(gcnt[g]), int(min_code[g])))
    osel = spec['osel']
    if osel is not None:
        kind = 'K' if osel == spec['rk_pi'] else spec['aggs'][osel][0]
        keyf = {'K': lambda r: r[0], 'AVG_LEN': lambda r: r[1],
                'COUNT_STAR': lambda r: r[2], 'MIN_DICT': lambda r: r[3]}[kind]
        rows.sort(key=keyf, reverse=True)
    if spec['lim'] is not None:
        rows = rows[spec['off']: spec['off'] + spec['lim']]
    out = []
    for lab, avg, cnt, mc in rows:
        r = [None] * len(spec['proj'])
        r[spec['rk_pi']] = lab
        for pi, (kind, _c) in spec['aggs'].items():
            r[pi] = avg if kind == 'AVG_LEN' else (cnt if kind == 'COUNT_STAR'
                     else wdb_sql._pyval(seg.fetch(col, mc)))
        out.append(tuple(r))
    _HITS += 1
    return out, [wdb_sql._alias(p) for p in spec['proj']]
