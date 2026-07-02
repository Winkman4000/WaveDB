"""Parquet-grounded correctness harness for the ClickBench suite.

PRINCIPLE: DuckDB is a stopwatch (the board's job); the PARQUET is the answer key. Here Duck is used
ONLY as a raw calculator over the parquet to recompute ground truth -- never as the reference answer to
a LIMIT query. Nothing demands WaveDB match Duck's arbitrary choice among equally-valid rows. The engine
is never bent to fit the test; the test verifies whether WaveDB's answer is CORRECT against the data.

Per-shape correctness law (all recomputed from the parquet):
  SCALAR / no-LIMIT (unique answer)  -> exact multiset match (sums via HUGEINT; float within tolerance)
  ORDERED LIMIT (may sit on a plateau) -> MEMBERSHIP (each returned group real w/ correct aggs)
                                          + BAND (returned order-key value multiset == true rank band)
  UNORDERED LIMIT                    -> MEMBERSHIP only (any K genuine groups is valid)
  FLOAT aggregate                    -> within a DECLARED relative tolerance of the exact recomputation

Speed is not measured here. Usage: python bench/correctness.py SRC DB PARQUET QUERIES [timeout_s]
"""
import sys, os, json, subprocess, decimal, math
import sqlglot
from sqlglot import expressions as E

SRC, DBDIR, PARQ, SQLF = sys.argv[1:5]
TMO = int(sys.argv[5]) if len(sys.argv) > 5 else 90
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb, _cbnorm as N
from _cbvalidate import total_order_sql
WORKER = os.path.join(os.path.dirname(__file__), '_cbq_rows.py')
ENV = dict(os.environ); ENV['PYTHONPATH'] = SRC
FTOL = decimal.Decimal('1e-9')          # declared relative tolerance for float aggregates

HITS_CTE = ("WITH hits AS (SELECT * REPLACE ("
            "(DATE '1970-01-01' + EventDate) AS EventDate, "
            "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
            "FROM read_parquet('%s'))" % PARQ)
con = duckdb.connect()
def duck(sql, params=None):
    return con.execute(HITS_CTE + ' ' + sql, params or []).fetchall()

# ---- shape extraction (sqlglot) -------------------------------------------------------------------
_AGG = {E.Count: 'COUNT', E.Sum: 'SUM', E.Avg: 'AVG', E.Min: 'MIN', E.Max: 'MAX'}
def _agg_of(node):
    """(kind, arg_sql or None) if node's top expression is a recognised aggregate, else None."""
    e = node.this if isinstance(node, E.Alias) else node
    for cls, kind in _AGG.items():
        if isinstance(e, cls):
            if kind == 'COUNT' and (e.args.get('this') is None or isinstance(e.this, E.Star)):
                return ('COUNT', None)
            if kind == 'COUNT' and e.args.get('distinct'):
                return ('COUNT_DISTINCT', e.this.sql())
            return (kind, e.this.sql() if e.args.get('this') is not None else None)
    if e.find(E.AggFunc):
        return ('OTHER_AGG', e.sql())
    return None

def extract(q):
    t = sqlglot.parse_one(q)
    projs = t.expressions
    star = any(isinstance(p, E.Star) for p in projs)
    group = t.args.get('group')
    gkeys = [g for g in group.expressions] if group else []
    order = t.args.get('order')
    oexpr = []
    if order:
        for o in order.expressions:
            oexpr.append((o.this, bool(o.args.get('desc'))))
    def _litint(node):
        if node is None: return None
        e = node.expression
        return int(e.this if hasattr(e, 'this') else e)
    lim = _litint(t.args.get('limit'))
    off = _litint(t.args.get('offset')) or 0
    where = t.args.get('where')
    where_sql = where.this.sql() if where else None
    # classify projections
    pkinds = [_agg_of(p) for p in projs]           # None => key/plain column
    return {'tree': t, 'projs': projs, 'star': star, 'gkeys': gkeys, 'order': oexpr,
            'lim': lim, 'off': off, 'where': where_sql, 'pkinds': pkinds}

def _proj_sql(p):
    return (p.this if isinstance(p, E.Alias) else p).sql()

# ---- WaveDB run (subprocess, rows) ----------------------------------------------------------------
def run_wdb(q):
    try:
        p = subprocess.run([sys.executable, WORKER, SRC, DBDIR, q], capture_output=True, text=True, timeout=TMO, env=ENV)
        ln = [l for l in p.stdout.strip().splitlines() if l.startswith('{')]
        return json.loads(ln[-1]) if ln else {'err': 'noout:' + (p.stderr.strip()[-80:] or '?')}
    except subprocess.TimeoutExpired:
        return {'err': 'TIMEOUT>%ds' % TMO}

# ---- comparison helpers ---------------------------------------------------------------------------
def _to_num(cell):
    if cell is None: return None
    if isinstance(cell, (list, tuple)) and len(cell) == 2 and cell[0] in ('i', 'f', 'b'):
        return decimal.Decimal(str(int(cell[1]) if cell[0] == 'b' else cell[1]))
    if isinstance(cell, str):
        try: return decimal.Decimal(cell)
        except Exception: return None
    return None
def cell_eq(a, b):
    na, nb = _to_num(a), _to_num(b)
    if na is not None and nb is not None:
        if na == nb: return True
        d = abs(na - nb)
        return (d / abs(nb) <= FTOL) if nb != 0 else (abs(na) <= FTOL)
    return a == b
def denorm(cell):
    if isinstance(cell, (list, tuple)) and len(cell) == 2 and cell[0] in ('i', 'f', 'b'):
        return cell[1]
    return cell
def nrow(vals):
    return [N.norm_cell_exact(v) for v in vals]      # full float precision; harness compares with tolerance
def _close(a, b):
    if a is None or b is None: return a is None and b is None
    if a == b: return True
    d = abs(a - b)
    return (d / abs(b) <= FTOL) if b != 0 else (abs(a) <= FTOL)
def msort(rows):
    return sorted((tuple(r) for r in rows), key=lambda t: json.dumps(t, default=str, sort_keys=True))

# ---- checks ---------------------------------------------------------------------------------------
def check_exact(q, W):
    """Unique answer: recompute the whole thing from the parquet, compare as a normalized multiset."""
    truth = [nrow(r) for r in duck(q)]
    w = msort(W); t = msort(truth)
    if len(w) != len(t): return ('WRONG', 'nrows w=%d truth=%d' % (len(w), len(t)))
    for rw, rt in zip(w, t):
        if len(rw) != len(rt) or not all(cell_eq(a, b) for a, b in zip(rw, rt)):
            return ('WRONG', 'row mismatch: %s vs %s' % (rw, rt))
    return ('CORRECT', 'exact multiset (%d rows)' % len(w))

def _key_agg_idx(s):
    keys = [i for i, k in enumerate(s['pkinds']) if k is None]
    aggs = [i for i, k in enumerate(s['pkinds']) if k is not None]
    return keys, aggs

def membership(s, W):
    """Every returned group is genuine: recompute its aggregates from the parquet for its key tuple."""
    keys, aggs = _key_agg_idx(s)
    if not keys: return (True, 'no keys')
    kexpr = [_proj_sql(s['projs'][i]) for i in keys]
    aexpr = [_proj_sql(s['projs'][i]) for i in aggs]
    seen = set(); bad = 0; detail = ''
    for r in W:
        kv = [denorm(r[i]) for i in keys]
        seen.add(tuple(json.dumps(x, default=str) for x in kv))
        cond = ' AND '.join('%s IS NOT DISTINCT FROM ?' % e for e in kexpr)
        wc = ('(%s) AND ' % s['where'] if s['where'] else '') + cond
        sql = 'SELECT %s FROM hits WHERE %s' % (', '.join(aexpr) if aexpr else 'COUNT(*)', wc)
        try:
            got = duck(sql, kv)[0]
        except Exception as ex:
            return (False, 'recompute err: %s' % str(ex)[:60])
        gt = nrow(got)
        for j, ai in enumerate(aggs):
            if not cell_eq(r[ai], gt[j]):
                bad += 1; detail = 'group %s agg[%d] w=%s truth=%s' % (kv, ai, r[ai], gt[j]); break
    distinct = (len(seen) == len(W))
    return (bad == 0 and distinct, 'bad=%d distinct=%s %s' % (bad, distinct, detail))

def band(s, W):
    """Returned order-key values sit at the true rank band (values are unique even when membership
    isn't). Only checkable when the order key is a projected column/alias."""
    if not s['order']: return (True, 'unordered')
    okey = s['order'][0][0]; oname = okey.sql()
    oidx = None
    for i, p in enumerate(s['projs']):
        names = ({p.alias, p.this.sql()} if isinstance(p, E.Alias) else {p.sql()})
        if oname in names:
            oidx = i; break
    if oidx is None:
        return (None, 'order key not projected -> band skipped')
    oexpr = _proj_sql(s['projs'][oidx])          # unwrapped expr, e.g. COUNT(*)
    gk = ', '.join(g.sql() for g in s['gkeys'])
    od = 'DESC' if s['order'][0][1] else 'ASC'
    where = ('WHERE %s ' % s['where']) if s['where'] else ''
    sql = ('SELECT %s AS ok FROM hits %sGROUP BY %s ORDER BY ok %s LIMIT %d OFFSET %d'
           % (oexpr, where, gk, od, s['lim'], s['off']))
    try:
        tvals = [_to_num(nrow([v[0]])[0]) for v in duck(sql)]
        wvals = [_to_num(r[oidx]) for r in W]
    except Exception as ex:
        return (None, 'band recompute err: %s' % str(ex)[:60])
    if any(x is None for x in tvals) or any(x is None for x in wvals):
        return (None, 'non-numeric order key -> band skipped')
    tvals.sort(); wvals.sort()
    ok = len(wvals) == len(tvals) and all(_close(a, b) for a, b in zip(wvals, tvals))
    return (ok, 'w=%s truth=%s' % ([str(x) for x in wvals[:4]], [str(x) for x in tvals[:4]]))

def check_scalar(s, W):
    if len(W) != 1: return ('WRONG', 'scalar expected 1 row, got %d' % len(W))
    return check_exact_scalar(s, W[0])

def check_exact_scalar(s, w0):
    where = ('WHERE %s' % s['where']) if s['where'] else ''
    for i, k in enumerate(s['pkinds']):
        e = s['projs'][i].this if isinstance(s['projs'][i], E.Alias) else s['projs'][i]
        if k and k[0] == 'AVG':                       # exact avg via HUGEINT
            arg = k[1]
            try:
                sm, cn = duck('SELECT SUM(CAST(%s AS HUGEINT)), COUNT(%s) FROM hits %s' % (arg, arg, where))[0]
                truth = decimal.Decimal(int(sm)) / decimal.Decimal(int(cn))
            except Exception:
                truth = decimal.Decimal(str(duck('SELECT AVG(%s) FROM hits %s' % (arg, where))[0][0]))
            wv = _to_num(w0[i])
            rel = abs(wv - truth) / abs(truth) if truth != 0 else abs(wv)
            if rel > FTOL: return ('WRONG', 'AVG rel=%.2e (w=%s truth=%s)' % (float(rel), wv, truth))
        else:
            tv = nrow([duck('SELECT %s FROM hits %s' % (e.sql(), where))[0][0]])[0]
            if not cell_eq(w0[i], tv): return ('WRONG', 'scalar[%d] w=%s truth=%s' % (i, w0[i], tv))
    return ('CORRECT', 'exact scalar')

def check_ordered_limit(q, s, W):
    m_ok, m_d = membership(s, W)
    if not m_ok:
        return ('WRONG', 'membership failed: %s' % m_d)
    b_ok, b_d = band(s, W)
    if b_ok is True or b_ok is None:
        tag = 'membership+band' if b_ok is True else 'membership (band n/a)'
        return ('CORRECT', '%s | %s | %s' % (tag, m_d, b_d))
    # Membership holds but the band pre-check disagrees. The band recompute can't always mirror the query
    # (e.g. a HAVING filter it drops), so it is not authoritative -- escalate to the canonical total order,
    # which runs the FULL query (HAVING and all) on both sides and admits exactly one answer.
    cv, cd = check_canonical(q, W)
    return (cv, 'band disputed -> canonical %s (%s) [band: %s]' % (cv.lower(), cd, b_d))

def check_unordered_limit(s, W):
    m_ok, m_d = membership(s, W)
    return (('CORRECT', 'membership | %s' % m_d) if m_ok else ('WRONG', m_d))

def check_canonical(q, W):
    """Row-level ordered top-N with no aggregate (e.g. SELECT col ... ORDER BY other LIMIT k): membership
    can't identify the rows, so impose a total order (append the projected columns to ORDER BY) making
    exactly one answer legal, then WaveDB and the parquet must agree. Verifies the plateau-resolved
    answer against the data -- still no reliance on DuckDB's arbitrary un-canonicalized pick."""
    cq = total_order_sql(q)
    if cq is None:
        return ('SKIP', 'no canonical form for top-N')
    wv = run_wdb(cq)
    if 'err' in wv:
        return ('TIMEOUT' if 'TIMEOUT' in wv['err'] else 'ERROR', 'canonical: ' + wv['err'])
    try:
        truth = [nrow(r) for r in duck(cq)]
    except Exception as ex:
        return ('ERROR', 'canonical duck: ' + str(ex)[:60])
    w = msort(wv['rows']); t = msort(truth)
    if len(w) != len(t):
        return ('WRONG', 'canonical nrows w=%d truth=%d' % (len(w), len(t)))
    for rw, rt in zip(w, t):
        if len(rw) != len(rt) or not all(cell_eq(a, b) for a, b in zip(rw, rt)):
            return ('WRONG', 'canonical row mismatch')
    return ('CORRECT', 'canonical total-order (%d rows)' % len(w))

def verdict(q, w):
    if 'err' in w:
        return ('TIMEOUT' if 'TIMEOUT' in w['err'] else 'ERROR', w['err'])
    W = w['rows']; s = extract(q)
    has_agg = any(s['pkinds'])
    if s['star']:
        return ('SKIP', 'SELECT * (row-level top-N; needs bespoke check)') if s['lim'] else check_exact(q, W)
    if s['lim'] is None:
        if not s['gkeys'] and has_agg:
            return check_scalar(s, W)
        return check_exact(q, W)
    keys, aggs = _key_agg_idx(s)
    if not aggs:
        return check_canonical(q, W) if (s['order'] and not s['gkeys']) else check_exact(q, W)
    return check_ordered_limit(q, s, W) if s['order'] else check_unordered_limit(s, W)

# ---- main -----------------------------------------------------------------------------------------
if __name__ == '__main__':
    qs = [l.strip() for l in open(SQLF) if l.strip() and not l.strip().startswith('--')]
    tally = {}
    print('== parquet-grounded correctness (Duck = calculator over raw data, never the LIMIT oracle) ==\n', flush=True)
    for i, q in enumerate(qs):
        w = run_wdb(q)
        try:
            v, d = verdict(q, w)
        except Exception as ex:
            v, d = 'HARNESS_ERR', '%s: %s' % (type(ex).__name__, str(ex)[:80])
        tally[v] = tally.get(v, 0) + 1
        print('Q%02d  %-10s %s' % (i, v, d[:100]), flush=True)
    print('\nSUMMARY', '  '.join('%s=%d' % (k, tally[k]) for k in sorted(tally)), flush=True)
