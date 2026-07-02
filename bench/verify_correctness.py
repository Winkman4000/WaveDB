"""Correctness gate for EVERY ClickBench query -- verified against the PARQUET, not against DuckDB's
arbitrary output. DuckDB is only a SQL executor over the parquet here (any correct engine would give the
same canonical answer), so its tie/order choices never enter the verdict.

Per query, the criterion is "is WaveDB's answer THE correct answer derivable from the data":
  - GROUP BY / ordered queries: impose a total order (append non-aggregate outputs to ORDER BY) so
    exactly ONE answer is legal, then WaveDB and the parquet ground truth must agree. Plateaus at the
    LIMIT boundary can't cause a disagreement because the canonical form has no arbitrary cutoff.
  - scalar / all-aggregate queries: compare the value within float tolerance (int/COUNT exact; AVG of
    huge magnitudes matched to ~12 sig figs by _cbnorm). Either the number is right or it isn't.
The base answer is checked against the parquet first (exact agreement = correct, cheap for the
deterministic majority); only a disagreement escalates to the canonical form, so a genuine bug is told
apart from an arbitrary tie pick. WaveDB runs in a kill-timeout subprocess; a query that doesn't finish
is a 'timeout' (a SPEED matter for the board), never a correctness verdict.

Usage: python bench/verify_correctness.py SRC DBDIR HITS_PARQUET QUERIES_SQL [timeout_s]
"""
import sys, subprocess, json, os
SRC, DBDIR, PARQ, SQLF = sys.argv[1:5]
T = int(sys.argv[5]) if len(sys.argv) > 5 else 120
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC); sys.path.insert(0, HERE)
import duckdb, _cbnorm as N, sqlglot
from sqlglot import expressions as E
from _cbvalidate import total_order_sql
CW = os.path.join(HERE, '_cbcorrect_worker.py')
HITS_CTE = ("WITH hits AS (SELECT * REPLACE ("
            "(DATE '1970-01-01' + EventDate) AS EventDate, "
            "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
            "FROM read_parquet('%s'))" % PARQ)
con = duckdb.connect()
env = dict(os.environ); env['PYTHONPATH'] = SRC

def duck_hash(q):
    return N.limit_hash(con.execute(HITS_CTE + ' ' + q).fetchall())

def wdb(q):
    try:
        p = subprocess.run([sys.executable, CW, SRC, DBDIR, q], capture_output=True, text=True, timeout=T, env=env)
        ln = [l for l in p.stdout.strip().splitlines() if l.startswith('{')]
        return json.loads(ln[-1]) if ln else {'err': 'noout:' + (p.stderr.strip()[-100:] or '?')}
    except subprocess.TimeoutExpired:
        return {'err': 'TIMEOUT>%ds' % T}

def classify(q):
    t = sqlglot.parse_one(q)
    if any(isinstance(p, E.Star) for p in t.expressions):
        return 'star'
    if total_order_sql(q) is not None:
        return 'ordered'
    return 'scalar'

def verify(q):
    # 1. Run the base query on WaveDB (kill-timeout) and compare to the parquet answer. Agreement with a
    #    correct engine's parquet computation confirms correctness cheaply -- no expensive canonical form
    #    for the deterministic majority. A base timeout is a SPEED verdict, never a correctness one.
    w = wdb(q)
    if 'err' in w:
        return ('timeout' if 'TIMEOUT' in w['err'] else 'error'), w['err'], 'base'
    try:
        dh = duck_hash(q)
    except Exception as e:
        return 'error', 'duck: ' + str(e)[:80], 'base'
    if w['hash'] == dh:
        return 'correct', '', 'exact agreement'
    # 2. Disagreement is NOT a verdict of wrong -- it may be a plateau/unordered LIMIT where DuckDB's
    #    arbitrary pick differs. Escalate to the canonical total order, where exactly one answer is legal,
    #    so a genuine bug is distinguishable from an arbitrary tie choice. Only a canonical miss is WRONG.
    kind = classify(q)
    if kind != 'ordered':
        return 'WRONG', 'base value differs beyond tolerance', kind  # scalar/star: deterministic, so real
    cq = total_order_sql(q)
    if cq is None:
        return 'WRONG', 'no canonical form', 'ordered'
    w2 = wdb(cq)
    if 'err' in w2:
        return ('timeout' if 'TIMEOUT' in w2['err'] else 'error'), w2['err'], 'canonical total-order'
    try:
        d2 = duck_hash(cq)
    except Exception as e:
        return 'error', 'duck: ' + str(e)[:80], 'canonical total-order'
    return ('correct' if w2['hash'] == d2 else 'WRONG'), '', 'canonical total-order'

qs = [l.strip() for l in open(SQLF) if l.strip() and not l.strip().startswith('--')]
print("== correctness gate: WaveDB vs canonical parquet ground truth (timeout=%ds) ==\n" % T, flush=True)
res = {}
for i, q in enumerate(qs):
    label = 'Q%02d' % i
    try:
        v, msg, how = verify(q)
    except Exception as e:
        v, msg, how = 'error', str(e)[:90], '?'
    res[label] = v
    print("  %s  %-8s  %-22s %s" % (label, v, how, ('<' + msg if msg else '')), flush=True)
from collections import Counter
c = Counter(res.values())
print("\nSUMMARY  correct=%d  WRONG=%d  timeout=%d  error=%d  (/%d)"
      % (c['correct'], c['WRONG'], c['timeout'], c['error'], len(qs)), flush=True)
wrong = [k for k, v in res.items() if v == 'WRONG']
print("  *** WRONG (real bugs): %s" % (', '.join(wrong) if wrong else 'none'), flush=True)
json.dump(res, open('/tmp/correctness.json', 'w'), indent=0)
