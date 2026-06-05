"""
Correctness MATRIX -- the same query battery run against many structurally-different
datasets, each compared to DuckDB (the oracle). The point: catch bugs that only appear
on a particular structure (mode-4 keys, nulls, empty tables, scattered order, high
cardinality, string vs int dims, float vs int measures, collisions). Generates one
discrete test per (variant x query) so the runner counts them individually.

Run standalone:  python3 tests/test_matrix.py            (full report)
                 python3 tests/test_matrix.py <substr>   (filter)
"""
import sys, os, tempfile, uuid, math
from decimal import Decimal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import duckdb, wdb_encode
from wdb_db import Database

# ----------------------------------------------------------------------------
# Variant specs. Every variant produces a table t(id, g, h, n, d, k) so one
# query battery applies to all. Knobs vary the STRUCTURE.
# ----------------------------------------------------------------------------
def spec(name, N=2000, cg=6, ch=4, cd=40, ck=9, ntype='float', gtype='str',
         nulls=(), order='id', nseg=1):
    return dict(name=name, N=N, cg=cg, ch=ch, cd=cd, ck=ck, ntype=ntype,
                gtype=gtype, nulls=tuple(nulls), order=order, nseg=nseg)

VARIANTS = [
    spec('base'),
    spec('mid', N=20000),
    spec('tiny', N=10),
    spec('one_row', N=1),
    spec('empty', N=0),
    spec('g_intdim', gtype='int'),
    spec('g_highcard', cg=1000),
    spec('g_unique', cg=2000),                 # cg==N -> every g distinct
    spec('g_constant', cg=1),                  # one group / huge collision
    spec('n_int', ntype='int'),
    spec('nulls_g', nulls=('g',)),
    spec('nulls_n', nulls=('n',)),
    spec('nulls_gn', nulls=('g', 'n')),
    spec('scattered', order='random'),
    spec('clustered_d', order='d'),
    spec('fine_dates', cd=1500),
    spec('big_k', ck=200),
    spec('h_wide', ch=37),
    spec('dup_heavy', cg=2, ch=2, ck=2),       # many identical rows
    spec('mid_scattered', N=20000, order='random'),
    # --- numeric edge structures ---
    spec('neg_measure', ntype='neg'),
    spec('zero_measure', ntype='zero'),
    spec('const_measure', ntype='const'),            # mode-6 synthetic const
    spec('neg_scattered', ntype='neg', order='random'),
    spec('nulls_n_int', ntype='int', nulls=('n',)),
    # --- null edge structures ---
    spec('allnull_g', gtype='null'),                 # exercises empty-dict guard
    spec('nulls_d', nulls=('d',)),                   # nullable datetime in group/filter
    spec('nulls_d_scattered', nulls=('d',), order='random'),
    # --- cardinality / mode structures ---
    spec('g_bool', cg=2),                            # boolean-like dim
    spec('k_seq', ck=5000),                          # k becomes 0..N-1 sequential -> mode-4
    spec('highcard_scattered', cg=1000, order='random'),
    # --- multi-segment (distinct merge code path) ---
    spec('two_seg', N=4000, nseg=2),
    spec('two_seg_scattered', N=4000, nseg=2, order='random'),
    spec('two_seg_clustered', N=4000, nseg=2, order='d'),
    spec('two_seg_nulls', N=4000, nseg=2, nulls=('g', 'n')),
    spec('three_seg', N=6000, nseg=3),
]


def _gen_sql(s):
    def nul(col, expr):
        return f"CASE WHEN (i%10)=0 THEN NULL ELSE {expr} END" if col in s['nulls'] else expr
    if s['gtype'] == 'null':
        g = "CAST(NULL AS VARCHAR)"
    else:
        g = nul('g', (f"'G'||(i%{s['cg']})" if s['gtype'] == 'str' else f"(i%{s['cg']})"))
    nmap = {'float': "((i%97)+1)*1.5", 'int': "((i%97)+1)",
            'neg': "((i%97)-48)*1.5", 'zero': "(i%5)", 'const': "7.0"}
    n = nul('n', nmap.get(s['ntype'], nmap['float']))
    h = f"(i%{s['ch']})"
    d = nul('d', f"(DATE '2020-01-01' + CAST(i%{s['cd']} AS INTEGER))")
    k = f"(i%{s['ck']})"
    return (f"SELECT CAST(i AS BIGINT) id, {g} AS g, {h} AS h, {n} AS n, "
            f"{d} AS d, {k} AS k FROM range({s['N']}) t(i)")


# generator-aware literals (valid for empty tables too; no need to sample data)
def _lits(s):
    geq = "'G0'" if s['gtype'] == 'str' else "0"
    g2 = "'G1'" if s['gtype'] == 'str' else "1"
    thr = max(1, s['N'] // (max(s['cg'], 1) * 4))
    return dict(geq=geq, g2=g2, thr=thr)


_WT = {'BIGINT': 'int', 'INTEGER': 'int', 'HUGEINT': 'int', 'VARCHAR': 'string',
       'DOUBLE': 'float', 'FLOAT': 'float', 'DATE': 'datetime', 'TIMESTAMP': 'datetime'}

_CACHE = {}
def _build(s):
    if s['name'] in _CACHE:
        return _CACHE[s['name']]
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS {_gen_sql(s)}")
    d = os.path.join(tempfile.gettempdir(), f"mx_{s['name']}_{uuid.uuid4().hex[:6]}")
    os.makedirs(d, exist_ok=True)
    desc = con.execute("DESCRIBE t").fetchall()
    sel = ", ".join((f"CAST({c[0]} AS DOUBLE) AS {c[0]}" if c[1].startswith('DECIMAL') else c[0])
                    for c in desc)
    order = {'id': 'id', 'd': 'd, id', 'random': 'hash(id)'}[s['order']]   # hash = deterministic shuffle
    db = Database.create(d)
    db.cat.add_table('t', [[c[0], ('float' if c[1].startswith('DECIMAL') else _WT[c[1]])] for c in desc])
    nseg = s.get('nseg', 1); N = s['N']
    per = (-(-N // nseg)) if nseg > 1 else N                        # ceil division
    for i in range(nseg):
        pq = os.path.join(d, f't_{i}.parquet')
        clause = f"LIMIT {per} OFFSET {i*per}" if nseg > 1 else ""
        con.execute(f"COPY (SELECT {sel} FROM t ORDER BY {order} {clause}) TO '{pq}' (FORMAT parquet)")
        wdb_encode.encode(pq, os.path.join(d, f't_{i}.wdb'))
        db.cat.add_segment('t', f't_{i}.wdb')
    _CACHE[s['name']] = (db, con, _lits(s))
    return _CACHE[s['name']]


# ----------------------------------------------------------------------------
# Query battery. Each: name -> (lits) -> (sql, ordered). Generator-aware literals
# keep them valid on every variant incl. empty.
# ----------------------------------------------------------------------------
QUERIES = [
    ('grp_g_count',   lambda L: ("SELECT g, COUNT(*) FROM t GROUP BY g", False)),
    ('grp_g_sum',     lambda L: ("SELECT g, SUM(n) FROM t GROUP BY g", False)),
    ('grp_g_avg',     lambda L: ("SELECT g, AVG(n) FROM t GROUP BY g", False)),
    ('grp_g_min',     lambda L: ("SELECT g, MIN(n) FROM t GROUP BY g", False)),
    ('grp_g_max',     lambda L: ("SELECT g, MAX(n) FROM t GROUP BY g", False)),
    ('grp_g_multi',   lambda L: ("SELECT g, COUNT(*), SUM(n), AVG(n), MIN(n), MAX(n) FROM t GROUP BY g", False)),
    ('grp_gh',        lambda L: ("SELECT g, h, COUNT(*), SUM(n) FROM t GROUP BY g, h", False)),
    ('grp_h_sum',     lambda L: ("SELECT h, SUM(n) FROM t GROUP BY h", False)),
    ('grp_k_sum',     lambda L: ("SELECT k, SUM(n) FROM t GROUP BY k", False)),
    ('grp_d_count',   lambda L: ("SELECT d, COUNT(*) FROM t GROUP BY d", False)),
    ('whole_agg',     lambda L: ("SELECT COUNT(*), SUM(n), AVG(n), MIN(n), MAX(n) FROM t", False)),
    ('where_n_sum',   lambda L: ("SELECT SUM(n) FROM t WHERE n > 50", False)),
    ('where_n_grp',   lambda L: ("SELECT g, SUM(n) FROM t WHERE n > 50 GROUP BY g", False)),
    ('where_geq_cnt', lambda L: (f"SELECT COUNT(*) FROM t WHERE g = {L['geq']}", False)),
    ('where_btw_n',   lambda L: ("SELECT COUNT(*), SUM(n) FROM t WHERE n BETWEEN 20 AND 80", False)),
    ('where_d_btw',   lambda L: ("SELECT COUNT(*) FROM t WHERE d BETWEEN DATE '2020-01-05' AND DATE '2020-01-10'", False)),
    ('where_in_g',    lambda L: (f"SELECT COUNT(*) FROM t WHERE g IN ({L['geq']}, {L['g2']})", False)),
    ('where_and_or',  lambda L: (f"SELECT SUM(n) FROM t WHERE n > 30 AND (g = {L['geq']} OR h = 1)", False)),
    ('where_k_eq',    lambda L: ("SELECT SUM(n) FROM t WHERE k = 3", False)),
    ('distinct_g',    lambda L: ("SELECT DISTINCT g FROM t", False)),
    ('distinct_gh',   lambda L: ("SELECT DISTINCT g, h FROM t", False)),
    ('distinct_k',    lambda L: ("SELECT DISTINCT k FROM t", False)),
    ('cdist_g',       lambda L: ("SELECT COUNT(DISTINCT g) FROM t", False)),
    ('cdist_k',       lambda L: ("SELECT COUNT(DISTINCT k) FROM t", False)),
    ('grp_cdist',     lambda L: ("SELECT h, COUNT(DISTINCT g) FROM t GROUP BY h", False)),
    ('order_limit',   lambda L: ("SELECT g, SUM(n) s FROM t GROUP BY g ORDER BY g LIMIT 3", True)),
    ('having',        lambda L: (f"SELECT g, COUNT(*) c FROM t GROUP BY g HAVING COUNT(*) > {L['thr']}", False)),
    ('is_null_g',     lambda L: ("SELECT COUNT(*) FROM t WHERE g IS NULL", False)),
    ('count_col',     lambda L: ("SELECT g, COUNT(n) FROM t GROUP BY g", False)),   # COUNT(col) skips nulls
]


# ----------------------------------------------------------------------------
# Compare vs DuckDB (the oracle). Round floats (small-N sums stay exact at 2dp),
# sort unless the query is ORDER BY.
# ----------------------------------------------------------------------------
def _sn(c):
    if isinstance(c, bool):
        return c
    if isinstance(c, int):
        return c
    if isinstance(c, (float, Decimal)):
        return round(float(c), 2)
    if c is None:
        return None
    return str(c)

def _norm(rows, ordered):
    out = [tuple(_sn(c) for c in r) for r in rows]
    return out if ordered else sorted(out, key=repr)

def _check_sql(s, sql, ordered):
    db, con, L = _build(s)
    got = _norm(db.run(sql)[0], ordered)
    exp = _norm([tuple(r) for r in con.execute(sql).fetchall()], ordered)
    assert got == exp, (f"[{s['name']}] {sql}\n got {got[:6]}\n exp {exp[:6]}")


def _mk(s, qf):
    def t():
        sql, ordered = qf(_build(s)[2])
        _check_sql(s, sql, ordered)
    return t

for _s in VARIANTS:
    for _qn, _qf in QUERIES:
        globals()[f"test_mx_{_s['name']}__{_qn}"] = _mk(_s, _qf)


if __name__ == '__main__':
    import traceback
    filt = sys.argv[1] if len(sys.argv) > 1 else ''
    names = sorted(n for n in dict(globals()) if n.startswith('test_mx_'))
    ok = fail = 0; fails = []
    for n in names:
        if filt and filt not in n:
            continue
        try:
            globals()[n](); ok += 1
        except Exception as e:
            fail += 1; fails.append((n, traceback.format_exc()))
    print(f"\n{ok} passed, {fail} failed (of {ok+fail} run)")
    for n, tb in fails[:25]:
        print(f"\n--- {n} ---\n{tb}")
