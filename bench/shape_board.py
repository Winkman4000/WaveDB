"""THE SHAPE BOARD: one ClickBench query shape, every column profile in every slot (the Q13 board's method,
2026-10-13, for any shape with two column slots).

Shapes (slot 1 = 'k', slot 2 = 'd' in the records):
  q13  SELECT K, COUNT(DISTINCT D) AS u FROM hits WHERE K <> <K's most common value> GROUP BY K
       ORDER BY u DESC LIMIT 10                                 (form 'open' drops the WHERE)
  q16  SELECT K1, K2, COUNT(*) FROM hits GROUP BY K1, K2 ORDER BY COUNT(*) DESC LIMIT 10
  q18  SELECT K1, extract(minute FROM EventTime) AS m, K2, COUNT(*) FROM hits GROUP BY K1, m, K2
       ORDER BY COUNT(*) DESC LIMIT 10
A profile is a column's passport (mode, code encoding, type) plus its V class (bench/q13_profiles.py); one
representative column stands for each profile (the original's own columns pinned), and the board runs every
ordered pair of representatives.

The target: every fill-in served by a read built for it (never general_scan), as close to the original's
submitted time as the data allows. Per pair we record the warm run, the hot run (best of the next two),
the winning read, the row count, DuckDB's time, and whether the answers agree (ties escalate to the
total-order form, as the shifting board does). A pair past --timeout is cut and recorded as TIMEOUT.

Resumable: pairs already in OUT/results.jsonl are skipped, so a restart continues the run.

Usage: python bench/shape_board.py SRC DB_DIR HITS_PARQUET PROFILES_JSON OUT_DIR --shape q18
       [--timeout 20] [--forms filtered,open] [--k COL,COL] [--d COL,COL] [--no-duck]
"""
import sys, os, json, time, argparse, threading, subprocess, selectors
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('db'); ap.add_argument('parquet'); ap.add_argument('profiles'); ap.add_argument('out')
ap.add_argument('--timeout', type=float, default=20.0)
ap.add_argument('--forms', default='')
ap.add_argument('--shape', default='q13')
ap.add_argument('--k', default='')
ap.add_argument('--d', default='')
ap.add_argument('--no-duck', action='store_true')
A = ap.parse_args()
sys.path.insert(0, A.src)
import duckdb, _cbnorm as N
from _cbvalidate import total_order_sql

# submitted times: benchmark/clickbench/results/20261006/c6a.4xlarge.json, [cold, hot1, hot2] -> cold, min(hot)
SHAPES = {
    'q13': {'orig': ('SearchPhrase', 'UserID'), 'cold': 0.766963, 'hot': 0.498798, 'forms': 'filtered'},
    'q16': {'orig': ('UserID', 'SearchPhrase'), 'cold': 0.695235, 'hot': 0.443119, 'forms': 'plain'},
    'q18': {'orig': ('UserID', 'SearchPhrase'), 'cold': 0.633505, 'hot': 0.343725, 'forms': 'plain'},
}
SH = SHAPES[A.shape]
ORIG_K, ORIG_D = SH['orig']
SUBMITTED = {'cold': SH['cold'], 'hot': SH['hot']}
MAX_ROWS = 10000
HITS_CTE = ("WITH hits AS (SELECT * REPLACE ("
            "(DATE '1970-01-01' + EventDate) AS EventDate, "
            "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
            "FROM read_parquet('%s'))" % A.parquet)
os.makedirs(A.out, exist_ok=True)
RES = os.path.join(A.out, 'results.jsonl')


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- the representatives

P = json.load(open(A.profiles))
COLS = {c['col']: c for c in P['columns']}


def clean(v):
    return isinstance(v, (int, float)) or (isinstance(v, str) and all(32 <= ord(ch) < 127 for ch in v))


def representative(members):
    for anchor in (ORIG_K, ORIG_D):
        if anchor in members:
            return anchor
    ok = [c for c in members if clean(COLS[c]['top']) and COLS[c]['top_share'] < 1.0]
    pool = ok or members
    return max(pool, key=lambda c: COLS[c]['V'])


REPS = {prof: representative(m) for prof, m in sorted(P['profiles'].items())}
PROF_OF = {c: p for p, c in REPS.items()}


def literal(col):
    c = COLS[col]; v = c['top']
    if c['dt'] in ('int', 'float') and not isinstance(v, str):
        return str(v)
    if c['dt'] in ('int', 'float'):
        return v
    return "'%s'" % str(v).replace("'", "''")


def sql_of(k, d, form):
    if A.shape == 'q16':
        return 'SELECT %s, %s, COUNT(*) FROM hits GROUP BY %s, %s ORDER BY COUNT(*) DESC LIMIT 10' % (k, d, k, d)
    if A.shape == 'q18':
        return ('SELECT %s, extract(minute FROM EventTime) AS m, %s, COUNT(*) FROM hits GROUP BY %s, m, %s '
                'ORDER BY COUNT(*) DESC LIMIT 10' % (k, d, k, d))
    where = '' if form == 'open' else ' WHERE %s <> %s' % (k, literal(k))
    return 'SELECT %s, COUNT(DISTINCT %s) AS u FROM hits%s GROUP BY %s ORDER BY u DESC LIMIT 10' % (k, d, where, k)


# ---------------------------------------------------------------- engines (as bench/shifting_board.py)

con = duckdb.connect()
con.execute('PRAGMA threads=%d' % int(os.environ.get('WDB_MAIN_THREADS', '16')))


def duck(sql, runs, timeout):
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        times = []; rows = None
        for _ in range(runs):
            t0 = time.perf_counter()
            r = con.execute(HITS_CTE + ' ' + sql).fetchmany(MAX_ROWS + 1)
            times.append(time.perf_counter() - t0)
            if rows is None:
                rows = r
        return rows, times
    finally:
        timer.cancel()


class Wave:
    def __init__(self):
        self.start()

    def start(self):
        env = dict(os.environ); env['PYTHONPATH'] = A.src
        self.p = subprocess.Popen([sys.executable, os.path.join(HERE, '_shift_worker.py'), A.src, A.db],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.p.stdout, selectors.EVENT_READ)
        self._read(600)

    def _read(self, timeout):
        if not self.sel.select(timeout):
            return None
        line = self.p.stdout.readline()
        return json.loads(line) if line else {'err': 'worker died'}

    def ask(self, sql, runs, timeout):
        self.p.stdin.write(json.dumps({'sql': sql, 'runs': runs}) + '\n'); self.p.stdin.flush()
        r = self._read(timeout * runs)
        if r is None:
            self.p.kill(); self.p.wait(); self.start()
            return {'err': 'TIMEOUT>%gs' % timeout}
        if r.get('err') == 'worker died':
            self.start()
        return r


wave = Wave()


def same_answer(sql, w_hash, d_rows):
    if w_hash == N.limit_hash(d_rows):
        return True
    cq = total_order_sql(sql)
    if cq is None:
        return False
    # the escalation is a check, not a timing: it gets the long cap (at 1x it cut slow-but-right
    # answers and logged them WRONG -- K=EventTime, 2026-10-13)
    w2 = wave.ask(cq, 1, max(90.0, A.timeout * 4))
    if 'err' in w2:
        return False
    d2, _ = duck(cq, 1, max(90.0, A.timeout * 4))
    return w2['hash'] == N.limit_hash(d2)


# ---------------------------------------------------------------- the run

done = set()
if os.path.exists(RES):
    for line in open(RES):
        try:
            r = json.loads(line); done.add((r['k'], r['d'], r['form']))
        except Exception:
            pass
LOG = open(RES, 'a')

ks = A.k.split(',') if A.k else list(REPS.values())
ds = A.d.split(',') if A.d else list(REPS.values())
pairs = [(ORIG_K, ORIG_D)] + [(k, d) for k in ks for d in ds if k != d and (k, d) != (ORIG_K, ORIG_D)]
forms = (A.forms or SH['forms']).split(',')
todo = [(k, d, f) for f in forms for k, d in pairs if (k, d, f) not in done]
say('profiles %d  representatives %d  pairs %d x forms %d  ->  %d to run (%d already done)  timeout %gs'
    % (len(P['profiles']), len(REPS), len(pairs), len(forms), len(todo), len(done), A.timeout))
say('shape %s  submitted: cold %.3fs  hot %.3fs' % (A.shape, SUBMITTED['cold'], SUBMITTED['hot']))
t_start = time.time()
for n, (k, d, form) in enumerate(todo, 1):
    sql = sql_of(k, d, form)
    rec = {'shape': A.shape, 'k': k, 'd': d, 'form': form, 'kprof': COLS[k]['profile'], 'dprof': COLS[d]['profile'],
           'sql': sql, 'sub_hot': SUBMITTED['hot']}
    w = wave.ask(sql, 1, A.timeout)
    if 'err' in w:
        rec.update(verdict='TIMEOUT' if w['err'].startswith('TIMEOUT') else 'error', err=w['err'], warm=None, hot=None)
    else:
        rec.update(warm=w['times'][0], read=w['read'], nrows=w['nrows'])
        w2 = wave.ask(sql, 2, A.timeout) if w['times'][0] < A.timeout / 2 else None
        rec['hot'] = min(w2['times']) if w2 and 'times' in w2 else w['times'][0]
        rec['x_submitted'] = round(rec['hot'] / SUBMITTED['hot'], 2)
        if not A.no_duck:
            try:
                d_rows, d_t = duck(sql, 2, max(60.0, A.timeout * 3))
                rec['duck_s'] = min(d_t)
                rec['verdict'] = 'correct' if same_answer(sql, w['hash'], d_rows) else 'WRONG'
            except Exception as e:
                rec['verdict'] = 'unchecked'; rec['duck_err'] = str(e)[:120]
    LOG.write(json.dumps(rec) + '\n'); LOG.flush()
    el = time.time() - t_start
    say('[%4d/%d %5.0fm] K=%-20s D=%-20s %-8s hot %8s  %-16s %s' % (
        n, len(todo), el / 60, k, d, form, ('%.3fs' % rec['hot']) if rec.get('hot') is not None else '-',
        rec.get('read', '-'), rec.get('verdict', '')))
say('Q13BOARD_DONE')                                   # (the run scripts wait on this word)
