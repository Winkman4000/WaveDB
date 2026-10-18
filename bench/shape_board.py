"""THE SHAPE BOARD: one ClickBench query shape, every column profile in every slot (the Q13 board's method,
2026-10-13, for any shape with two column slots).

Shapes (slot 1 = 'k', slot 2 = 'd' in the records):
  q13  SELECT K, COUNT(DISTINCT D) AS u FROM hits WHERE K <> <K's most common value> GROUP BY K
       ORDER BY u DESC LIMIT 10                                 (form 'open' drops the WHERE)
  q16  SELECT K1, K2, COUNT(*) FROM hits GROUP BY K1, K2 ORDER BY COUNT(*) DESC LIMIT 10
  q18  SELECT K1, extract(minute FROM EventTime) AS m, K2, COUNT(*) FROM hits GROUP BY K1, m, K2
       ORDER BY COUNT(*) DESC LIMIT 10
  q31  SELECT K1, K2, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> ''
       GROUP BY K1, K2 ORDER BY c DESC LIMIT 10    (Q30/Q31; form 'open' drops the WHERE = Q32)
  q27  SELECT K, AVG(length(S)) AS l, COUNT(*) AS c FROM hits WHERE S <> '' GROUP BY K HAVING COUNT(*) > 100000
       ORDER BY l DESC LIMIT 25      (slot 2 = S, text columns only; form 'open' drops the WHERE)
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
       [--duck-cache FILE] [--engines both|duck|wave]

The DuckDB cache holds, per query text, DuckDB's answer hash and its times tagged with the machine
(CPU model x cores). An answer is the same on any machine, so it is reused everywhere; a time is reused
only on the machine that measured it.
  --engines both   (default) WaveDB and DuckDB; DuckDB runs only for what the cache lacks on this machine
  --engines duck   DuckDB alone (no WaveDB worker): fill the cache with every pair's answer AND its
                   total-order (tie-break) answer, so a later WaveDB-only run never needs DuckDB
  --engines wave   WaveDB alone: never touches DuckDB or the parquet; answers come from the cache, a pair
                   whose answer is missing is recorded 'uncached' and listed at the end
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
ap.add_argument('--duck-cache', default='', help="JSON lines: DuckDB's answer hash and times per query, reused "
                "on later runs (answers reused on any machine, times only on the machine that measured them)")
ap.add_argument('--engines', default='both', choices=('both', 'duck', 'wave', 'route'))
ap.add_argument('--hot-runs', type=int, default=2, help='hot runs after the warm one (best kept); 1 for a census')
ap.add_argument('--pairs', default='', help='a samples.json (strains.py): run only these (k, d, form) queries')
ap.add_argument('--no-escalate', action='store_true', help="a hash mismatch is recorded tie_unchecked instead of "
                "running the total-order form on WaveDB (minutes when every count ties); legal_check.py decides it")
ap.add_argument('--wait-answers', type=float, default=0.0, help="WaveDB-only: wait up to this many seconds per "
                "query for the DuckDB pod to save its answer (a board trailing a running --engines duck)")
A = ap.parse_args()
sys.path.insert(0, A.src)
import duckdb, _cbnorm as N
from _cbvalidate import total_order_sql

# submitted times: benchmark/clickbench/results/20261006/c6a.4xlarge.json, [cold, hot1, hot2] -> cold, min(hot)
SHAPES = {
    'q13': {'orig': ('SearchPhrase', 'UserID'), 'cold': 0.766963, 'hot': 0.498798, 'forms': 'filtered'},
    'q16': {'orig': ('UserID', 'SearchPhrase'), 'cold': 0.695235, 'hot': 0.443119, 'forms': 'plain'},
    'q18': {'orig': ('UserID', 'SearchPhrase'), 'cold': 0.633505, 'hot': 0.343725, 'forms': 'plain'},
    # ClickBench Q30/Q31 (filtered) and Q32 (open); submitted times from results/20261002_run6 (Q31, Q32)
    'q31': {'orig': ('WatchID', 'ClientIP'), 'pins': ('WatchID', 'ClientIP', 'SearchEngineID'),
            'cold': 0.771010, 'hot': 0.364645, 'forms': 'filtered,open',
            'by_form': {'filtered': (0.771010, 0.364645), 'open': (0.713697, 0.373296)}},
    # the census shapes (2026-10-17); submitted times read from results/20261002_run6 by query number.
    # Single-slot shapes (one column) run each representative once; 'd' is recorded as '-'.
    'q11': {'orig': ('MobilePhone', 'MobilePhoneModel'), 'qnum': 11, 'forms': 'filtered'},
    'q12': {'orig': ('SearchPhrase',), 'qnum': 12, 'forms': 'filtered'},
    'q14': {'orig': ('SearchEngineID', 'SearchPhrase'), 'qnum': 14, 'forms': 'filtered'},
    'q15': {'orig': ('UserID',), 'pins': ('UserID', 'URL'), 'qnum': 15, 'forms': 'plain'},
    'q35': {'orig': ('ClientIP',), 'qnum': 35, 'forms': 'plain', 'numeric': True},
    'q33': {'orig': ('URL',), 'qnum': 33, 'forms': 'plain'},
    'q34': {'orig': ('URL',), 'qnum': 34, 'forms': 'plain'},
    'q27': {'orig': ('CounterID', 'URL'), 'qnum': 27, 'forms': 'filtered,open', 'text_d': True},
}
SH = SHAPES[A.shape]
if 'qnum' in SH:
    try:
        _r = json.load(open(os.path.join(HERE, '..', 'benchmark', 'clickbench', 'results', '20261002_run6',
                                         'c6a.4xlarge.json')))['result'][SH['qnum']]
        SH['cold'], SH['hot'] = _r[0], min(_r[1:])
    except Exception:
        SH['cold'] = SH['hot'] = float('nan')
SINGLE = len(SH['orig']) == 1
ORIG_K, ORIG_D = (SH['orig'][0], '-') if SINGLE else SH['orig']
SUBMITTED = {'cold': SH['cold'], 'hot': SH['hot']}


def submitted_hot(form):
    return SH.get('by_form', {}).get(form, (SH['cold'], SH['hot']))[1]
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
    for anchor in SH.get('pins', SH['orig']):
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
    if A.shape == 'q11':
        return ('SELECT %s, %s, COUNT(DISTINCT UserID) AS u FROM hits WHERE %s <> %s GROUP BY %s, %s '
                'ORDER BY u DESC LIMIT 10' % (k, d, d, literal(d), k, d))
    if A.shape == 'q14':
        return ('SELECT %s, %s, COUNT(*) AS c FROM hits WHERE %s <> %s GROUP BY %s, %s ORDER BY c DESC LIMIT 10'
                % (k, d, d, literal(d), k, d))
    if A.shape == 'q12':
        return 'SELECT %s, COUNT(*) AS c FROM hits WHERE %s <> %s GROUP BY %s ORDER BY c DESC LIMIT 10' % (
            k, k, literal(k), k)
    if A.shape == 'q15':
        return 'SELECT %s, COUNT(*) FROM hits GROUP BY %s ORDER BY COUNT(*) DESC LIMIT 10' % (k, k)
    if A.shape == 'q33':
        return 'SELECT %s, COUNT(*) AS c FROM hits GROUP BY %s ORDER BY c DESC LIMIT 10' % (k, k)
    if A.shape == 'q27':
        where = " WHERE %s <> ''" % d if form == 'filtered' else ''
        return ('SELECT %s, AVG(length(%s)) AS l, COUNT(*) AS c FROM hits%s GROUP BY %s HAVING COUNT(*) > 100000 '
                'ORDER BY l DESC LIMIT 25' % (k, d, where, k))
    if A.shape == 'q34':
        return 'SELECT 1, %s, COUNT(*) AS c FROM hits GROUP BY 1, %s ORDER BY c DESC LIMIT 10' % (k, k)
    if A.shape == 'q35':
        return ('SELECT %s, %s - 1, %s - 2, %s - 3, COUNT(*) AS c FROM hits GROUP BY %s, %s - 1, %s - 2, %s - 3 '
                'ORDER BY c DESC LIMIT 10' % ((k,) * 8))
    if A.shape == 'q31':
        where = " WHERE SearchPhrase <> ''" if form == 'filtered' else ''
        return ('SELECT %s, %s, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits%s GROUP BY %s, %s '
                'ORDER BY c DESC LIMIT 10' % (k, d, where, k, d))
    where = '' if form == 'open' else ' WHERE %s <> %s' % (k, literal(k))
    return 'SELECT %s, COUNT(DISTINCT %s) AS u FROM hits%s GROUP BY %s ORDER BY u DESC LIMIT 10' % (k, d, where, k)


# ---------------------------------------------------------------- engines (as bench/shifting_board.py)

_con = []


def duck_con():
    """DuckDB opened on first use only, so a WaveDB-only run never starts it"""
    if A.engines == 'wave':
        raise RuntimeError('DuckDB asked for in a WaveDB-only run')
    if not _con:
        c = duckdb.connect()
        c.execute('PRAGMA threads=%d' % int(os.environ.get('WDB_MAIN_THREADS', '16')))
        _con.append(c)
    return _con[0]


def machine_tag():
    model = 'cpu?'
    try:
        for l in open('/proc/cpuinfo'):
            if l.startswith('model name'):
                model = l.split(':', 1)[1].strip(); break
    except Exception:
        pass
    return '%s x%d' % (model, len(os.sched_getaffinity(0)))


MACHINE = machine_tag()
_ANS = {}                     # sql -> DuckDB's answer hash (any machine)
_TIM = {}                     # sql -> DuckDB's times measured on THIS machine


_LOADED = [0.0]


def load_cache():
    _LOADED[0] = time.time()
    _ANS.clear(); _TIM.clear()
    if not (A.duck_cache and os.path.exists(A.duck_cache)):
        return
    for l in open(A.duck_cache):
        try:
            r = json.loads(l)
        except Exception:
            continue
        _ANS[r['sql']] = r['hash']
        if r.get('machine') == MACHINE and len(r.get('times', ())) > len(_TIM.get(r['sql'], ())):
            _TIM[r['sql']] = r['times']


load_cache()


def keep(sql, h, times):
    _ANS[sql] = h; _TIM[sql] = times
    if A.duck_cache:
        with open(A.duck_cache, 'a') as f:
            f.write(json.dumps({'sql': sql, 'hash': h, 'times': times, 'machine': MACHINE}) + '\n')


def duck_answer(sql, timeout):
    """DuckDB's answer hash: the cache (any machine), else run it once (None in a WaveDB-only run)"""
    if sql not in _ANS and A.engines == 'wave':
        # the DuckDB pod may have added it since we started; with --wait-answers, trail it and wait
        t_end = time.time() + A.wait_answers
        while True:
            if time.time() - _LOADED[0] > (10 if A.wait_answers else 60):
                load_cache()
            if sql in _ANS or time.time() >= t_end:
                break
            time.sleep(5)
    if sql in _ANS:
        return _ANS[sql]
    if A.engines == 'wave':
        return None
    rows, times = duck(sql, 1, timeout)
    keep(sql, N.limit_hash(rows), times)
    return _ANS[sql]


def duck_cached(sql, runs, timeout):
    """(DuckDB's answer hash, its times on this machine): run only for what the cache lacks here"""
    if len(_TIM.get(sql, ())) >= runs:
        return _ANS[sql], _TIM[sql]
    rows, times = duck(sql, runs, timeout)
    keep(sql, N.limit_hash(rows), times)
    return _ANS[sql], times


def duck(sql, runs, timeout):
    con = duck_con()
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
        # THE WORKER DIES WITH THE BOARD (2026-10-17): a board killed mid-query had left its worker running
        # the query on 13 of 16 cores for 3 h, and every board after it was timed on what was left
        import atexit, signal
        atexit.register(self.stop)
        for sg in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sg, lambda *_a: (self.stop(), os._exit(1)))

    def stop(self):
        try:
            self.p.kill(); self.p.wait(5)
        except Exception:
            pass

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


wave = Wave() if A.engines in ('both', 'wave') else None
LONG = max(90.0, A.timeout * 4)   # the escalation is a check, not a timing: it gets the long cap (at 1x it
                                  # cut slow-but-right answers and logged them WRONG -- K=EventTime, 2026-10-13)


def same_answer(sql, w_hash, d_hash):
    """'correct' / 'WRONG' / 'uncached' (a WaveDB-only run whose tie-break answer is not in the cache)"""
    if w_hash == d_hash:
        return 'correct'
    if A.no_escalate:
        # a census leaves every mismatch to legal_check.py (an exact legality check on the DuckDB pod)
        return 'tie_unchecked'
    cq = total_order_sql(sql)
    if cq is None:
        return 'WRONG'
    d2 = duck_answer(cq, LONG)
    if d2 is None:
        return 'uncached'
    w2 = wave.ask(cq, 1, LONG)
    if 'err' in w2:
        # the tie-break form did not finish: the answer is UNCHECKED, not wrong (2026-10-17: WatchID x
        # JavaEnable, every count 1, the total-order form ran past the cap and had been logged WRONG)
        return 'tie_unchecked' if str(w2['err']).startswith('TIMEOUT') else 'WRONG'
    return 'correct' if w2['hash'] == d2 else 'WRONG'


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
if SH.get('text_d'):                         # slot 2 takes a string function: text columns only
    ds = [c for c in ds if COLS[c]['dt'] == 'str']
if SH.get('numeric'):
    ks = [c for c in ks if COLS[c]['dt'] == 'int' or c == ORIG_K]
if SINGLE:
    pairs = [(ORIG_K, '-')] + [(k, '-') for k in ks if k != ORIG_K]
else:
    pairs = [(ORIG_K, ORIG_D)] + [(k, d) for k in ks for d in ds if k != d and (k, d) != (ORIG_K, ORIG_D)]
forms = (A.forms or SH['forms']).split(',')
todo = [(k, d, f) for f in forms for k, d in pairs if (k, d, f) not in done]
if A.pairs:                                  # THE STRAIN SAMPLE: a handful per strain instead of the grid
    todo = [(x['k'], x['d'], x['form']) for x in json.load(open(A.pairs)) if (x['k'], x['d'], x['form']) not in done]
say('profiles %d  representatives %d  pairs %d x forms %d  ->  %d to run (%d already done)  timeout %gs'
    % (len(P['profiles']), len(REPS), len(pairs), len(forms), len(todo), len(done), A.timeout))
say('shape %s  submitted: cold %.3fs  hot %.3fs' % (A.shape, SUBMITTED['cold'], SUBMITTED['hot']))
say('engines %s  machine %s  cache: %d answers, %d timed here' % (A.engines, MACHINE, len(_ANS), len(_TIM)))
t_start = time.time()

if A.engines == 'route':
    # THE ROUTE MAP (2026-10-17, Jackson's "failing is passing"): no query is answered. Every read's execute is
    # replaced by a marker that stops at the first read whose detect claims the query -- covered, in
    # milliseconds; a query that reaches the general scan (or the new doors) is a HOLE. Blind spots: a read
    # that claims and then hands on while executing (tripletop's empty bar) counts as covered here.
    os.environ['WDB_PLANCACHE_OFF'] = '1'
    import read_methods as RM, controller as CT, wdb_db as WD

    class _Claimed(BaseException):         # past every "except Exception" fallback inside run()
        pass

    _SPEC = [None]

    def _marker(nm):
        def f(*_a, **_k):
            _SPEC[0] = _a[1] if len(_a) > 1 else None   # execute(ctx, spec): the read's own plan, for the strain
            raise _Claimed(nm)
        return f

    def _spec_summary(sp):
        # the plan's scalar fields only (which key rides low, the filter, bit widths) -- what picks a branch
        if not isinstance(sp, dict):
            return None
        return {k9: v9 for k9, v9 in sp.items() if isinstance(v9, (str, int, float, bool)) or v9 is None}
    for _nm in dir(RM):
        _o = getattr(RM, _nm)
        if isinstance(_o, RM.Read):
            _o.execute = _marker(_o.name)
    RM.general_scan = _marker('general_scan')
    WD.Database._new_doors = lambda self, *a, **k: (_ for _ in ()).throw(_Claimed('new_doors'))
    rdb = WD.Database.open(A.db)
    holes = []
    counts = {}
    for n, (k, d, form) in enumerate(todo, 1):
        sql = sql_of(k, d, form)
        rec = {'shape': A.shape, 'k': k, 'd': d, 'form': form, 'kprof': COLS[k]['profile'],
               'dprof': COLS[d]['profile'] if d in COLS else '-', 'sql': sql}
        t0 = time.perf_counter()
        try:
            rdb.run(sql)
            rec['read'] = '(answered outside the controller)'
        except _Claimed as e:
            rec['read'] = str(e)
            rec['spec'] = _spec_summary(_SPEC[0])
        except Exception as e:
            rec['read'] = 'error'; rec['err'] = ('%s: %s' % (type(e).__name__, e))[:300]
        _SPEC[0] = None
        rec['route_s'] = round(time.perf_counter() - t0, 4)
        rec['hole'] = rec['read'] in ('general_scan', 'new_doors', 'error', '(answered outside the controller)')
        rec['verdict'] = 'HOLE' if rec['hole'] else 'routed'
        if rec['hole']:
            holes.append((k, d, form, rec['read']))
        counts[rec['read']] = counts.get(rec['read'], 0) + 1
        LOG.write(json.dumps(rec) + '\n'); LOG.flush()
    say('ROUTE %s: %d queries, %d holes  reads %s' % (A.shape, len(todo), len(holes),
        ' '.join('%s=%d' % kv for kv in sorted(counts.items(), key=lambda kv: -kv[1]))))
    for h in holes[:40]:
        say('  HOLE %s x %s %s -> %s' % h)
    say('Q13BOARD_DONE')
    sys.exit(0)

if A.engines == 'duck':
    # DuckDB alone: every pair's answer and its tie-break answer into the cache (resumable through the cache)
    n_run = 0
    for n, (k, d, form) in enumerate(todo, 1):
        sql = sql_of(k, d, form); cq = total_order_sql(sql)
        t0 = time.time(); state = 'cached'
        try:
            if sql not in _ANS or (cq is not None and cq not in _ANS):
                duck_answer(sql, max(60.0, A.timeout * 3))
                if cq is not None:
                    duck_answer(cq, LONG)
                state = 'run'; n_run += 1
        except Exception as e:
            state = 'ERR ' + str(e)[:80]
        say('[%4d/%d %5.0fm] K=%-20s D=%-20s %-8s %-6s %.2fs' % (
            n, len(todo), (time.time() - t_start) / 60, k, d, form, state, time.time() - t0))
    say('duck: %d pairs run, %d already cached' % (n_run, len(todo) - n_run))
    say('Q13BOARD_DONE')
    sys.exit(0)

uncached = []
holes = []
for n, (k, d, form) in enumerate(todo, 1):
    sql = sql_of(k, d, form)
    rec = {'shape': A.shape, 'k': k, 'd': d, 'form': form, 'kprof': COLS[k]['profile'],
           'dprof': COLS[d]['profile'] if d in COLS else '-',
           'sql': sql, 'sub_hot': submitted_hot(form)}
    w = wave.ask(sql, 1, A.timeout)
    if 'err' in w:
        rec.update(verdict='TIMEOUT' if w['err'].startswith('TIMEOUT') else 'error', err=w['err'], warm=None, hot=None)
    else:
        rec.update(warm=w['times'][0], read=w['read'], nrows=w['nrows'])
        w2 = wave.ask(sql, A.hot_runs, A.timeout) if A.hot_runs > 0 and w['times'][0] < A.timeout / 2 else None
        rec['hot'] = min(w2['times']) if w2 and 'times' in w2 else w['times'][0]
        rec['x_submitted'] = round(rec['hot'] / submitted_hot(form), 2)
        if not A.no_duck:
            try:
                if A.engines == 'wave':
                    d_hash = duck_answer(sql, 0)
                    if len(_TIM.get(sql, ())) >= 2:            # DuckDB timed on this machine before
                        rec['duck_s'] = min(_TIM[sql])
                else:
                    d_hash, d_t = duck_cached(sql, 2, max(60.0, A.timeout * 3))
                    rec['duck_s'] = min(d_t)
                rec['verdict'] = 'uncached' if d_hash is None else same_answer(sql, w['hash'], d_hash)
                if rec['verdict'] == 'uncached':
                    uncached.append(sql)
            except Exception as e:
                rec['verdict'] = 'unchecked'; rec['duck_err'] = str(e)[:120]
        if rec.get('read') == 'general_scan':
            # STRICT (2026-10-17): a pair the general path answers is a HOLE in the grid, however right
            # its answer -- the board counts it as a failure, never as covered
            rec['hole'] = True; holes.append((k, d, form))
    LOG.write(json.dumps(rec) + '\n'); LOG.flush()
    el = time.time() - t_start
    say('[%4d/%d %5.0fm] K=%-20s D=%-20s %-8s hot %8s  %-16s %s' % (
        n, len(todo), el / 60, k, d, form, ('%.3fs' % rec['hot']) if rec.get('hot') is not None else '-',
        rec.get('read', '-'), rec.get('verdict', '') + (' HOLE' if rec.get('hole') else '')))
say('HOLES %d of %d run (general_scan answered)%s' % (len(holes), len(todo),
    (': ' + ' | '.join('%s x %s %s' % h for h in holes[:30])) if holes else ''))
if uncached:
    say('UNCACHED %d (run --engines duck for these, then re-run): %s' % (len(uncached), ' | '.join(uncached[:20])))
say('Q13BOARD_DONE')                                 # (the run scripts wait on this word)
