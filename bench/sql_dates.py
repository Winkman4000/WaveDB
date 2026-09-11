"""THE DATES PROBE: a 2M-row realm with real DATE and TIMESTAMP columns, and the
date/time constructs analysts write, vs duck. Classifies OK/WRONG/HOLE/CRASH.
usage: python3 bench/sql_dates.py gen | python3 bench/sql_dates.py run
"""
import sys, os, time, json, signal
sys.path.insert(0, 'src')
import numpy as np

ROOT = '/workspace/data/dates'
DB = ROOT + '/db'
N = 2_000_000

def gen():
    import pyarrow as pa, pyarrow.parquet as pq
    import wdb_encode
    os.makedirs(DB, exist_ok=True)
    rng = np.random.default_rng(11)
    days = rng.integers(0, 3650, N) + np.datetime64('2015-01-01').astype('datetime64[D]').astype(np.int64)
    d = days.astype('datetime64[D]')
    secs = rng.integers(0, 86400, N)
    ts = (days.astype('datetime64[D]').astype('datetime64[s]').astype(np.int64) + secs).astype('datetime64[s]').astype('datetime64[us]')
    t = pa.table({
        'id': pa.array(np.arange(N, dtype=np.int64)),
        'k': pa.array(rng.integers(1, 51, N).astype(np.int32)),
        'amount': pa.array(np.round(rng.random(N) * 1000, 2)),
        'd': pa.array(d), 'ts': pa.array(ts),
        'd2': pa.array((days + rng.integers(1, 400, N)).astype('datetime64[D]')),
        'name': pa.array(['n%03d' % v for v in rng.integers(0, 500, N)]),
    })
    p = ROOT + '/ev.parquet'; pq.write_table(t, p, row_group_size=1_000_000)
    t0 = time.perf_counter(); wdb_encode.encode(p, DB + '/ev_0.wdb')
    cat = {'tables': {'ev': {'segments': ['ev_0.wdb'], 'columns': t.column_names,
                             'schema': [['id', 'int'], ['k', 'int'], ['amount', 'float'], ['d', 'date'], ['ts', 'timestamp'], ['d2', 'date'], ['name', 'str']]}}}
    json.dump(cat, open(DB + '/catalog.json', 'w'))
    print('dates realm: %d rows encoded in %.0fs' % (N, time.perf_counter() - t0), flush=True)

Q = [
 ('cmp', 'date_gt_literal',   "SELECT COUNT(*) FROM ev WHERE d >= DATE '2020-06-01'"),
 ('cmp', 'date_between',      "SELECT COUNT(*) FROM ev WHERE d BETWEEN DATE '2018-01-01' AND DATE '2018-12-31'"),
 ('cmp', 'ts_gt_literal',     "SELECT COUNT(*) FROM ev WHERE ts >= TIMESTAMP '2021-03-04 12:00:00'"),
 ('cmp', 'date_eq',           "SELECT COUNT(*) FROM ev WHERE d = DATE '2019-07-04'"),
 ('cmp', 'date_in',           "SELECT COUNT(*) FROM ev WHERE d IN (DATE '2019-07-04', DATE '2020-01-01')"),
 ('cmp', 'date_lt_string',    "SELECT COUNT(*) FROM ev WHERE d < '2016-01-01'"),
 ('ext', 'extract_year',      "SELECT EXTRACT(YEAR FROM d) AS y, COUNT(*) FROM ev GROUP BY y ORDER BY y"),
 ('ext', 'extract_month',     "SELECT EXTRACT(MONTH FROM d) AS m, SUM(amount) FROM ev GROUP BY m ORDER BY m"),
 ('ext', 'extract_dow',       "SELECT EXTRACT(DOW FROM d) AS w, COUNT(*) FROM ev GROUP BY w ORDER BY w"),
 ('ext', 'extract_doy',       "SELECT COUNT(*) FROM ev WHERE EXTRACT(DOY FROM d) = 100"),
 ('ext', 'extract_hour_ts',   "SELECT EXTRACT(HOUR FROM ts) AS h, COUNT(*) FROM ev GROUP BY h ORDER BY h"),
 ('ext', 'year_fn',           "SELECT YEAR(d) AS y, COUNT(*) FROM ev WHERE YEAR(d) = 2017 GROUP BY y"),
 ('ext', 'month_day_fn',      "SELECT COUNT(*) FROM ev WHERE MONTH(d) = 2 AND DAY(d) = 29"),
 ('ext', 'date_part',         "SELECT date_part('quarter', d) AS q, COUNT(*) FROM ev GROUP BY q ORDER BY q"),
 ('trunc', 'trunc_month',     "SELECT DATE_TRUNC('month', d) AS m, COUNT(*) FROM ev GROUP BY m ORDER BY m LIMIT 5"),
 ('trunc', 'trunc_year',      "SELECT DATE_TRUNC('year', d) AS y, SUM(amount) FROM ev GROUP BY y ORDER BY y"),
 ('trunc', 'trunc_week',      "SELECT COUNT(DISTINCT DATE_TRUNC('week', d)) FROM ev"),
 ('trunc', 'trunc_day_ts',    "SELECT DATE_TRUNC('day', ts) AS dd, COUNT(*) FROM ev WHERE ts >= TIMESTAMP '2024-12-25 00:00:00' GROUP BY dd ORDER BY dd"),
 ('arith', 'plus_interval',   "SELECT COUNT(*) FROM ev WHERE d + INTERVAL 30 DAY < d2"),
 ('arith', 'minus_interval',  "SELECT COUNT(*) FROM ev WHERE d2 - INTERVAL 1 YEAR > d"),
 ('arith', 'date_diff_days',  "SELECT AVG(d2 - d) FROM ev"),
 ('arith', 'datediff_fn',     "SELECT MAX(DATEDIFF('day', d, d2)) FROM ev"),
 ('arith', 'date_diff_month', "SELECT COUNT(*) FROM ev WHERE DATE_DIFF('month', d, d2) >= 6"),
 ('arith', 'cast_ts_date',    "SELECT COUNT(*) FROM ev WHERE CAST(ts AS DATE) = d"),
 ('arith', 'cast_str_date',   "SELECT COUNT(*) FROM ev WHERE d > CAST('2022-01-01' AS DATE)"),
 ('agg', 'min_max_date',      "SELECT MIN(d), MAX(d), MIN(ts), MAX(ts) FROM ev"),
 ('agg', 'count_distinct_date', "SELECT COUNT(DISTINCT d) FROM ev"),
 ('agg', 'group_date',        "SELECT d, COUNT(*) AS c FROM ev GROUP BY d ORDER BY c DESC, d LIMIT 3"),
 ('agg', 'order_by_date',     "SELECT id, d FROM ev WHERE k = 7 ORDER BY d, id LIMIT 5"),
 ('agg', 'first_last_day',    "SELECT COUNT(*) FROM ev WHERE d = LAST_DAY(d)"),
 ('fmt', 'strftime',          "SELECT STRFTIME(d, '%Y-%m') AS ym, COUNT(*) FROM ev GROUP BY ym ORDER BY ym LIMIT 3"),
 ('fmt', 'epoch',             "SELECT MAX(EPOCH(ts)) FROM ev"),
 ('fmt', 'to_string_cast',    "SELECT COUNT(*) FROM ev WHERE CAST(d AS VARCHAR) LIKE '2019-0%'"),
 ('win', 'lag_date',          "SELECT id, d, LAG(d) OVER (ORDER BY id) AS prev FROM ev WHERE k = 3 AND id < 100000 ORDER BY id LIMIT 5"),
 ('join', 'self_range',       "SELECT COUNT(*) FROM ev a JOIN ev b ON a.id = b.id - 1 WHERE b.d > a.d AND a.k = 1 AND b.k = 1"),
]

def run():
    import duckdb
    import wdb_kernels; wdb_kernels.warm()
    from wdb_db import Database
    db = Database.open(DB)
    con = duckdb.connect(); con.execute("CREATE VIEW ev AS SELECT * FROM read_parquet('%s/ev.parquet')" % ROOT)
    import datetime
    def norm(v):
        if v is None: return 'NULL'
        if isinstance(v, bool): return 'b:%d' % v
        if isinstance(v, float): return '%.6g' % v
        if isinstance(v, datetime.datetime): return v.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(v, datetime.date): return v.strftime('%Y-%m-%d')
        if isinstance(v, np.datetime64): return str(v.astype('datetime64[s]')).replace('T', ' ') if 'T' in str(v) else str(v)
        return str(v)
    def same(w, e):
        return len(w) == len(e) and sorted(tuple(norm(v) for v in r) for r in w) == sorted(tuple(norm(v) for v in r) for r in e)
    def _alarm(sig, frm): raise TimeoutError('timeout')
    signal.signal(signal.SIGALRM, _alarm)
    tally = {}
    for cat, name, q in Q:
        try:
            e = con.execute(q).fetchall()
        except Exception as ex:
            print('%-6s %-20s DUCK-ERR %s' % (cat, name, str(ex)[:60]), flush=True); continue
        signal.alarm(120)
        try:
            w = db.run(q); w = w[0] if isinstance(w, tuple) else w; signal.alarm(0)
            st = 'OK' if same(w, e) else 'WRONG'
            detail = '' if st == 'OK' else ' wave=%s duck=%s' % (str(w[:2])[:50], str(e[:2])[:50])
        except NotImplementedError as ex:
            signal.alarm(0); st = 'HOLE'; detail = ' ' + str(ex)[:70]
        except BaseException as ex:
            signal.alarm(0); st = 'CRASH'; detail = ' %s: %s' % (type(ex).__name__, str(ex)[:60])
        tally[st] = tally.get(st, 0) + 1
        print('%-6s %-20s %-5s%s' % (cat, name, st, detail), flush=True)
    print('DATES: %d constructs | %s' % (len(Q), ' '.join('%s=%d' % kv for kv in sorted(tally.items()))), flush=True)

if __name__ == '__main__':
    gen() if sys.argv[1] == 'gen' else run()
