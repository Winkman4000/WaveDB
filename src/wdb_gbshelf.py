"""wdb_gbshelf -- Jackson's tiered-absence count shelf (.gbc2).

The census of per-code row counts, dressed by its own histogram:
count==1 stored as NOTHING (absence from every structure IS the answer),
count==2 as one presence bit, count==3 as one presence bit, count>=4 as a
presence bit plus a rank-ordered u16 tail with 512-code popcount
checkpoints. mmap-shaped: opening costs microseconds (the pickle it
replaces paid 65ms per query to answer one entry). Point lookups are three
independent bit tests plus at most one tail touch; bulk rebuilds are three
vectorized unpacks and one scatter.
"""
import os
import mmap
import numpy as np

_MAGIC = 0x67626332          # 'gbc2'
_CK = 512                    # checkpoint every 512 codes


def _path(seg, col):
    return seg.path + '.%s.gbc2' % col


def birth(seg, col):
    """Build the shelf from one bincount of the raw code stream. Returns True
    on success. Birth-on-first-touch: pays the pooled read once."""
    try:
        import wdb_sidecar, os as _os9
        if not wdb_sidecar.births_on(_os9.path.dirname(seg.path)):
            return False                     # THE SWITCH: a disk-only shelf is not born
        c = seg.cols[col]
        V = int(c['V'])
        cnt = np.bincount(np.asarray(seg._raw_codes(col)), minlength=V)
        if cnt.size and int(cnt.max()) >= 65536:
            return False                     # u16 tail law: this column outgrew it
        bm2 = np.packbits(cnt == 2)
        bm3 = np.packbits(cnt == 3)
        b4 = cnt >= 4
        bm4 = np.packbits(b4)
        nb = (V + 7) // 8
        # exclusive cumulative popcount of bm4 per _CK-code block
        pop8 = np.unpackbits(bm4[:nb]).reshape(-1, 8).sum(1)
        blocks = (V + _CK - 1) // _CK
        bp = np.zeros(blocks, np.int64)
        per = _CK // 8
        s = np.add.reduceat(pop8, np.arange(0, pop8.size, per))
        bp[:s.size] = s
        ck4 = np.concatenate([[0], np.cumsum(bp)[:-1]]).astype(np.uint32)
        tail = cnt[b4].astype(np.uint16)
        hdr = np.array([_MAGIC, V, tail.size, int(seg.N)], np.int64)
        with open(_path(seg, col), 'wb') as f:
            f.write(hdr.tobytes())
            f.write(bm2[:nb].tobytes())
            f.write(bm3[:nb].tobytes())
            f.write(bm4[:nb].tobytes())
            f.write(ck4.tobytes())
            f.write(tail.tobytes())
        import wdb_shelves
        wdb_shelves.record(seg, 'gbc2', col=col)
        return True
    except Exception:
        return False


def open_shelf(seg, col):
    """mmap the shelf; returns views dict or None. No caching needed: opening
    is priced in microseconds by design."""
    p = _path(seg, col)
    if not os.path.exists(p):
        return None
    try:
        f = open(p, 'rb')
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        hdr = np.frombuffer(mm, np.int64, 4)
        if int(hdr[0]) != _MAGIC or int(hdr[3]) != int(seg.N):
            return None
        V = int(hdr[1]); T = int(hdr[2])
        nb = (V + 7) // 8
        blocks = (V + _CK - 1) // _CK
        o = 32
        bm2 = np.frombuffer(mm, np.uint8, nb, o); o += nb
        bm3 = np.frombuffer(mm, np.uint8, nb, o); o += nb
        bm4 = np.frombuffer(mm, np.uint8, nb, o); o += nb
        ck4 = np.frombuffer(mm, np.uint32, blocks, o); o += 4 * blocks
        tail = np.frombuffer(mm, np.uint16, T, o)
        return {'V': V, 'bm2': bm2, 'bm3': bm3, 'bm4': bm4,
                'ck4': ck4, 'tail': tail, '_mm': mm, '_f': f}
    except Exception:
        return None


def _bit(bm, code):
    return (bm[code >> 3] >> (7 - (code & 7))) & 1


def point(sh, code):
    """count for one code: three independent bit tests, one tail touch."""
    if _bit(sh['bm4'], code):
        blk = code >> 9
        base = int(sh['ck4'][blk])
        start = blk << 6                      # first byte of this block
        endb = code >> 3
        seg9 = sh['bm4'][start:endb]
        r = int.from_bytes(seg9.tobytes(), 'big').bit_count() if seg9.size else 0
        last = int(sh['bm4'][endb]) >> (7 - (code & 7))
        r += bin(last).count('1') - 1         # bits at/above ours, minus ours
        return int(sh['tail'][base + r])
    if _bit(sh['bm3'], code):
        return 3
    if _bit(sh['bm2'], code):
        return 2
    return 1                                  # absence IS the answer


def bulk(sh):
    """Full census, vectorized: ones + tier bits + tail scatter."""
    V = sh['V']
    cnt = np.ones(V, np.int64)
    cnt += np.unpackbits(sh['bm2'])[:V]
    cnt += 2 * np.unpackbits(sh['bm3'])[:V]
    idx = np.flatnonzero(np.unpackbits(sh['bm4'])[:V])
    cnt[idx] = sh['tail']
    return cnt


# ---- the dense census (gbc3): small-V columns whose counts outgrow the
# u16 tail law. V*8 bytes on disk, mmap-shaped, birthed from one bincount.

_MAGIC3 = 0x67626333


def _path3(seg, col):
    return seg.path + '.%s.gbc3' % col


def birth_dense(seg, col):
    try:
        import wdb_sidecar, os as _os9
        if not wdb_sidecar.births_on(_os9.path.dirname(seg.path)):
            return False                     # THE SWITCH
        c = seg.cols[col]
        V = int(c['V'])
        if V > 4096:
            return False
        cnt = np.bincount(np.asarray(seg._raw_codes(col)), minlength=V).astype(np.int64)
        hdr = np.array([_MAGIC3, V, 0, int(seg.N)], np.int64)
        with open(_path3(seg, col), 'wb') as f:
            f.write(hdr.tobytes())
            f.write(cnt.tobytes())
        import wdb_shelves
        wdb_shelves.record(seg, 'gbc3', col=col)
        return True
    except Exception:
        return False


def open_dense(seg, col):
    p = _path3(seg, col)
    if not os.path.exists(p):
        return None
    try:
        f = open(p, 'rb')
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        hdr = np.frombuffer(mm, np.int64, 4)
        if int(hdr[0]) != _MAGIC3 or int(hdr[3]) != int(seg.N):
            return None
        V = int(hdr[1])
        return {'cnt': np.frombuffer(mm, np.int64, V, 32), '_mm': mm, '_f': f}
    except Exception:
        return None


# ---- the gbcount lane: SELECT c, COUNT(*) FROM t [WHERE c <op> lit]
# GROUP BY c [ORDER BY ...] [LIMIT/OFFSET]. The census answers; predicates
# on the group key itself mask codes -- zero row reads.

def detect_group(ctx):
    try:
        import sqlglot.expressions as E
        t = ctx.tree
        if t.args.get('joins') or t.args.get('having') or t.args.get('distinct'):
            return None
        g = t.args.get('group')
        if not g or len(g.expressions) != 1:
            return None
        gexpr = g.expressions[0]
        if not isinstance(gexpr, E.Column):
            return None
        col = ctx.cmap.get(gexpr.name, gexpr.name)
        c9 = ctx.seg.cols.get(col)
        if not c9 or c9.get('has_null'):
            return None
        if c9.get('mode', 0) != 0:
            return None                      # the census speaks dictionary only
        if ctx.seg._effective(col) is not None:
            return None                      # overrides falsify the census
        if col in getattr(ctx.seg, '_synth', {}):
            return None                      # synthetic columns: no raw stream
        exprs = t.expressions
        if len(exprs) != 2:
            return None
        colpos = None
        cntpos = None
        for i, e in enumerate(exprs):
            b = e.this if isinstance(e, E.Alias) else e
            if isinstance(b, E.Column) and ctx.cmap.get(b.name, b.name) == col:
                colpos = i
            elif isinstance(b, E.Count) and isinstance(b.this, E.Star):
                cntpos = i
        if colpos is None or cntpos is None:
            return None
        w = t.args.get('where')
        pred = None
        if w is not None:
            e = w.this
            if isinstance(e, (E.NEQ, E.EQ)) and isinstance(e.this, E.Column) \
                    and ctx.cmap.get(e.this.name, e.this.name) == col \
                    and isinstance(e.expression, (E.Literal, E.Neg)):
                lv = e.expression.sql()
                pred = ('neq' if isinstance(e, E.NEQ) else 'eq', lv)
            else:
                return None
        o = t.args.get('order')
        oby = None
        if o is not None:
            if len(o.expressions) != 1:
                return None
            oe = o.expressions[0]
            base = oe.this
            desc = bool(oe.args.get('desc'))
            if isinstance(base, E.Count) or (isinstance(base, E.Column)
                                             and base.name.lower() in ('count', 'c', 'pageviews')
                                             and not isinstance(base, E.Column)) :
                oby = ('cnt', desc)
            elif isinstance(base, E.Column) and ctx.cmap.get(base.name, base.name) == col:
                oby = ('key', desc)
            elif isinstance(base, E.Column):
                al = exprs[cntpos]
                if isinstance(al, E.Alias) and al.alias == base.name:
                    oby = ('cnt', desc)
                else:
                    return None
            else:
                return None
        lim = t.args.get('limit')
        off = t.args.get('offset')
        k = int(lim.expression.sql()) if lim is not None else None
        of = int(off.expression.sql()) if off is not None else 0
        return {'col': col, 'colpos': colpos, 'pred': pred, 'order': oby,
                'k': k, 'off': of}
    except Exception:
        return None


def execute_group(ctx, spec):
    try:
        seg = ctx.seg
        col = spec['col']
        V = int(seg.cols[col]['V'])
        cnt = None
        if V <= 4096:
            sh = open_dense(seg, col)
            if sh is None:
                if not birth_dense(seg, col):
                    return None
                sh = open_dense(seg, col)
            if sh is None:
                return None
            cnt = np.array(sh['cnt'])
        else:
            sh = open_shelf(seg, col)
            if sh is None:
                if not birth(seg, col):
                    return None
                sh = open_shelf(seg, col)
            if sh is None:
                return None
            cnt = bulk(sh)
        codes = np.arange(V, dtype=np.int64)
        if spec['pred'] is not None:
            op, lv = spec['pred']
            try:
                val = lv.strip("'")
                tv = seg.fetch(col, 0)
                if isinstance(tv, (bytes, bytearray)):
                    lit = val.encode('utf-8')
                elif isinstance(tv, str):
                    lit = val
                else:
                    lit = int(val)
            except Exception:
                lit = lv
            import wdb_funnel
            pc = wdb_funnel._code_of(seg, col, lit)
            if op == 'neq':
                if pc is not None:
                    m = codes != pc
                    codes = codes[m]
                    cnt = cnt[m]
            else:
                if pc is None:
                    codes = codes[:0]
                    cnt = cnt[:0]
                else:
                    codes = codes[pc:pc + 1]
                    cnt = cnt[pc:pc + 1]
        if spec['order'] is not None:
            by, desc = spec['order']
            if by == 'cnt':
                o = np.argsort(-cnt if desc else cnt, kind='stable')
            else:
                o = np.argsort(-codes if desc else codes, kind='stable')
            codes = codes[o]
            cnt = cnt[o]
        a = spec['off']
        b = a + spec['k'] if spec['k'] is not None else codes.size
        codes = codes[a:b]
        cnt = cnt[a:b]
        rows = []
        for c9, n9 in zip(codes.tolist(), cnt.tolist()):
            v9 = seg.fetch(col, int(c9))
            if isinstance(v9, (bytes, bytearray)):
                v9 = v9.decode('utf-8', 'replace')
            r = [None, None]
            r[spec['colpos']] = v9
            r[1 - spec['colpos']] = int(n9)
            rows.append(tuple(r))
        import wdb_sql
        return rows, [wdb_sql._alias(e) for e in ctx.tree.expressions]
    except Exception:
        if os.environ.get('WDB_GBC_DEBUG'):
            import traceback
            traceback.print_exc()
        return None
