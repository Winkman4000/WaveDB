"""megaboard runner: 103 queries x (wave, duck) with per-kind validation.

Usage (pod): python3 bench/megaboard.py /workspace/data/fjdb /workspace/data/hits.parquet
Emits one line per query as it lands (nohup-friendly), then the summary block.
"""
import sys, time, os
sys.path.insert(0, 'src'); sys.path.insert(0, 'bench')
import sqlglot, sqlglot.expressions as E
import duckdb
import _cbnorm as N
from _cbvalidate import total_order_sql
from wdb_db import Database
import wdb_kernels
import wdb_colresult as CR

FJ = sys.argv[1] if len(sys.argv) > 1 else '/workspace/data/fjdb'
PQ = sys.argv[2] if len(sys.argv) > 2 else '/workspace/data/hits.parquet'
from megaboard_queries import QUERIES

wdb_kernels.warm()
db = Database.open(FJ)
con = duckdb.connect()
import pandas as pd
con.register('rdim', pd.read_parquet(os.path.join(FJ, 'dim.parquet')))
con.register('gdim', pd.read_parquet(os.path.join(FJ, 'gdim.parquet')))
CTE = ("WITH hits AS (SELECT * REPLACE ((DATE '1970-01-01'+EventDate) AS EventDate,"
       "(TIMESTAMP '1970-01-01'+to_seconds(EventTime)) AS EventTime)"
       " FROM read_parquet('%s', file_row_number=true))" % PQ)


def duck_cols(con, sql):
    """Duck's side of the columnar treaty: fetchnumpy in the timed lane -- the same
    transcription toll WaveDB stopped paying. ColRows wraps the arrays; validators
    materialize lazily (untimed) and see exactly the tuples they always saw. Falls
    back to fetchall for types fetchnumpy can't carry."""
    cur = con.execute(sql)
    try:
        dnp = cur.fetchnumpy()
        arrs = list(dnp.values())
        n = int(arrs[0].shape[0]) if arrs else 0
        return CR.ColRows([('arr', a) for a in arrs], n)
    except Exception:
        rows = cur.fetchall() if cur.description else []
        return rows


def rows_of(r):
    return r[0] if isinstance(r, tuple) else r


def duck_sql(q):
    if q.lstrip().upper().startswith('WITH '):
        return CTE + ', ' + q.lstrip()[5:]
    return CTE + ' ' + q


def wtie_duck_sql(q):
    """Append the parquet row number as the FINAL tiebreak inside every window ORDER.
    Window queries whose frames straddle order ties have no unique SQL answer (duck
    itself wobbles run-to-run); WaveDB's walk order IS parquet order, so pinning duck
    to (order cols, file_row_number) makes duck deterministic AND makes it compute
    the exact answer WaveDB's stable tie behavior already produces. A match proves
    the answer; the original 'S' compare against dice was a coin flip."""
    t = sqlglot.parse_one(q)
    for w in t.find_all(E.Window):
        o = w.args.get('order')
        if o is not None:
            o.set('expressions', list(o.expressions)
                  + [E.Ordered(this=E.column('file_row_number'))])
    return duck_sql(t.sql())


def order_col_index(q, hdr):
    t = sqlglot.parse_one(q)
    o = t.args.get('order')
    if o is None or not hdr:
        return None
    oe = o.expressions[0].this
    if not isinstance(oe, E.Column):
        return None
    low = [str(h).lower() for h in hdr]
    return low.index(oe.name.lower()) if oe.name.lower() in low else None


def norm(x):
    if hasattr(x, 'item'):
        x = x.item()
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, (int, float)):
        return round(float(x), 4)        # int-vs-float emission parity across engines
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        s = str(x)
        if s.endswith(' 00:00:00'):
            s = s[:-9]           # datetime64[us] vs date: same day, one dress code
        return s


def validate(kind, q, w, hdr, d):
    if kind == 'N':
        return len(w) == len(d)
    if kind == 'C':
        if len(w) != len(d):
            return False
        return all(norm(a) == norm(b) for ra, rb in zip(w, d) for a, b in zip(ra, rb))
    if kind == 'S':
        return (len(w) == len(d)
                and sorted(tuple(norm(x) for x in r) for r in w)
                == sorted(tuple(norm(x) for x in r) for r in d))
    if kind == 'W':
        # window-tie: duck rerun pinned to parquet row order (the order WaveDB walks);
        # full multiset compare against a now-deterministic oracle
        d2 = con.execute(wtie_duck_sql(q)).fetchall()
        return (len(w) == len(d2)
                and sorted(tuple(norm(x) for x in r) for r in w)
                == sorted(tuple(norm(x) for x in r) for r in d2))
    if kind == 'M':
        if len(w) != len(d):
            return False
        idx = order_col_index(q, hdr)
        if idx is None:
            idx = -1
        return sorted(norm(r[idx]) for r in w) == sorted(norm(r[idx]) for r in d)
    if kind == 'H':
        q2 = total_order_sql(q) or q
        wh = N.limit_hash(rows_of(db.run(q2)))
        dh = N.limit_hash(con.execute(duck_sql(q2)).fetchall())
        return wh == dh
    return False


results = []
for entry in QUERIES:
    name, kind, q = entry[0], entry[1], entry[2]
    q_duck = entry[3] if len(entry) > 3 else q      # optional duck-side override (frame tiebreaks)
    try:
        t = time.perf_counter(); w = rows_of(db.run(q)); w1 = time.perf_counter() - t
        out = db.run(q)
        t = time.perf_counter(); rows_of(db.run(q)); w2 = time.perf_counter() - t
        hdr = out[1] if isinstance(out, tuple) else None
        wt = min(w1, w2)
    except Exception as e:
        print('%-16s WAVE-ERROR %s' % (name, str(e)[:90]), flush=True)
        results.append((name, None, None, False)); continue
    try:
        dq = duck_sql(q_duck)
        t = time.perf_counter(); d = duck_cols(con, dq); d1 = time.perf_counter() - t
        t = time.perf_counter(); duck_cols(con, dq); d2 = time.perf_counter() - t
        dt = min(d1, d2)
    except Exception as e:
        print('%-16s DUCK-ERROR %s' % (name, str(e)[:90]), flush=True)
        results.append((name, wt, None, False)); continue
    try:
        ok = validate(kind, q, w, hdr, d)
    except Exception as e:
        print('%-16s VAL-ERROR %s' % (name, str(e)[:90]), flush=True)
        ok = False
    ratio = dt / wt if wt > 0 else 0
    results.append((name, wt, dt, ok))
    print('%-16s %s wave=%7.2fs duck=%7.2fs x%5.2f rows=%d' %
          (name, 'OK   ' if ok else 'FALSE', wt, dt, ratio, len(w)), flush=True)

good = [r for r in results if r[3] and r[1] and r[2]]
false_n = sum(1 for r in results if not r[3])
faster = sum(1 for r in good if r[1] < r[2])
ratios = sorted(r[2] / r[1] for r in good)
med = ratios[len(ratios) // 2] if ratios else 0
print('=' * 64, flush=True)
print('MEGABOARD: %d queries | ok=%d false/err=%d | wave faster on %d | median ratio %.2fx'
      % (len(results), len(good), false_n, faster, med), flush=True)
print('total wave %.0fs | total duck %.0fs'
      % (sum(r[1] for r in good), sum(r[2] for r in good)), flush=True)
