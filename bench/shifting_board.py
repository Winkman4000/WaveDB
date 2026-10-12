"""THE SHIFTING BOARD: the benchmark's queries, changed at random every run, so no read can be cut
to a query it will see -- only general code passes.

Per run (a fresh seed unless --seed is given; the seed is logged, so any finding replays exactly):
  1. Each ClickBench query gets K random variants (bench/_shift_mutate.py): a column swapped for one
     of the same kind, a literal or LIKE pattern drawn from the real data, another time unit,
     another aggregate, a condition added or dropped, a group key added or dropped, flipped
     comparisons, other LIMITs and sort directions. Plus the fixed set (bench/shifting_fixed.sql):
     every variant that ever exposed a problem, so a fixed cliff can't return unnoticed.
  2. A variant counts only if DuckDB answers it within the timeout with at most MAX_ROWS rows.
  3. CORRECTNESS (the gate): WaveDB's full answer must equal DuckDB's; a disagreement escalates to
     the total-order form (ties can't decide it), as bench/verify_correctness.py does.
  4. SPEED (the detector): the variant's EXPECTED WaveDB time is the original's, scaled by how much
     DuckDB slowed down for the same change. The variant is cut off at its BOUND -- 2x expected
     (3x below 0.1 s, where noise is loud; never under 0.25 s) -- because any overrun is the finding
     and its size only measures the distance between a fast path and the general one. An overrun
     needs two observations (the first run may pay a compile, so it gets a looser cap) before it is
     a cliff. cliff = (wave variant / wave original) / (duck variant / duck original); for a cut
     variant it is a lower bound. The winning read is shown for the original and the variant.

Usage: python bench/shifting_board.py SRC DB_DIR HITS_PARQUET OUT_DIR [--seed N] [--per-query K]
       [--only 13,18] [--timeout S] [--no-fixed]
"""
import sys, os, json, time, random, math, argparse, threading, subprocess, selectors, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('db'); ap.add_argument('parquet'); ap.add_argument('out')
ap.add_argument('--seed', type=int, default=None)
ap.add_argument('--per-query', type=int, default=3)
ap.add_argument('--only', default='')
ap.add_argument('--timeout', type=float, default=120.0)
ap.add_argument('--no-fixed', action='store_true')
A = ap.parse_args()
sys.path.insert(0, A.src)
import duckdb, _cbnorm as N, _shift_mutate as M
from _cbvalidate import total_order_sql

ROOT = os.path.dirname(HERE)
QUERIES = os.path.join(ROOT, 'benchmark/clickbench/queries.sql')
CREATE = os.path.join(ROOT, 'benchmark/clickbench/create.sql')
FIXED = os.path.join(HERE, 'shifting_fixed.sql')
MAX_ROWS = 10000
BOUND = 2.0            # a variant may take this many times its expected time ...
BOUND_SMALL = 3.0      # ... or this many when the expected time is under SMALL_S (noise is loud there)
SMALL_S = 0.1
MIN_BOUND_S = 0.25     # never cut a variant faster than this (ms jitter is not a path)
WARM_ALLOW = 4.0       # the first run may pay a numba compile: cap it at WARM_ALLOW x bound ...
WARM_MIN_S = 10.0      # ... but never under this
EPS = 0.01             # ClickBench's own +10 ms
HITS_CTE = ("WITH hits AS (SELECT * REPLACE ("
            "(DATE '1970-01-01' + EventDate) AS EventDate, "
            "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
            "FROM read_parquet('%s'))" % A.parquet)

seed = A.seed if A.seed is not None else int.from_bytes(os.urandom(4), 'little')
OUT = os.path.join(A.out, 'seed_%d_%s' % (seed, time.strftime('%Y%m%d_%H%M%S')))   # a rerun never overwrites
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'results.jsonl'), 'w')


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- engines

con = duckdb.connect()
con.execute('PRAGMA threads=%d' % int(os.environ.get('WDB_MAIN_THREADS', '16')))


def duck(sql, runs, timeout):
    """(rows, [times]) or raises; the query is interrupted at the timeout."""
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        times = []
        rows = None
        for _ in range(runs):
            t0 = time.perf_counter()
            cur = con.execute(HITS_CTE + ' ' + sql)
            r = cur.fetchmany(MAX_ROWS + 1)
            times.append(time.perf_counter() - t0)
            if rows is None:
                rows = r
        return rows, times
    finally:
        timer.cancel()


class Wave:
    def __init__(self):
        self.p = None
        self.start()

    def start(self):
        env = dict(os.environ)
        env['PYTHONPATH'] = A.src
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
        self.p.stdin.write(json.dumps({'sql': sql, 'runs': runs}) + '\n')
        self.p.stdin.flush()
        r = self._read(timeout * runs)
        if r is None:
            self.p.kill(); self.p.wait()
            self.start()
            return {'err': 'TIMEOUT>%ds' % timeout}
        if r.get('err') == 'worker died':
            self.start()
        return r


wave = Wave()


def hot(times):
    return min(times[1:]) if len(times) > 1 else times[0]


# ---------------------------------------------------------------- the data the variants draw from

def load_pool(schema):
    path = os.path.join(A.out, 'pool.json')
    if os.path.exists(path):
        return json.load(open(path))
    say('sampling values from the data (once; cached in %s) ...' % path)
    cols = sorted(schema)
    rows = con.execute(HITS_CTE + ' SELECT %s FROM hits USING SAMPLE 4000 ROWS' % ', '.join(cols)).fetchall()
    pool = {}
    for i, c in enumerate(cols):
        vals = []
        for r in rows:
            v = r[i]
            if isinstance(v, (bytes, bytearray)):
                v = v.decode('utf-8', 'replace')
            elif hasattr(v, 'isoformat'):
                v = v.isoformat(sep=' ') if hasattr(v, 'hour') else v.isoformat()
            vals.append(v)
        pool[c] = vals
    json.dump(pool, open(path, 'w'))
    return pool


schema = M.schema_from_create(CREATE)
pool = load_pool(schema)
lowcard = sorted(c for c in schema if 1 < len(set(pool[c])) <= 40)
qs = [l.strip().rstrip(';') for l in open(QUERIES) if l.strip()]
only = {int(x) for x in A.only.split(',') if x.strip()}
scope = [i for i in range(len(qs)) if not only or i in only]
say('seed %d  |  %d queries x %d variants  |  low-cardinality keys: %d' % (seed, len(scope), A.per_query, len(lowcard)))


# ---------------------------------------------------------------- correctness

def same_answer(sql, w_hash, d_rows):
    if w_hash == N.limit_hash(d_rows):
        return True
    cq = total_order_sql(sql)
    if cq is None:
        return False
    w2 = wave.ask(cq, 1, A.timeout)
    if 'err' in w2:
        return False
    d2, _ = duck(cq, 1, A.timeout)
    return w2['hash'] == N.limit_hash(d2)


def emit(rec):
    LOG.write(json.dumps(rec) + '\n')
    LOG.flush()


# ---------------------------------------------------------------- originals

base = {}
for i in scope:
    d_rows, d_t = duck(qs[i], 4, A.timeout)
    w = wave.ask(qs[i], 4, A.timeout)
    rec = {'kind': 'original', 'base': i, 'sql': qs[i], 'duck_s': hot(d_t)}
    if 'err' in w:
        rec.update(verdict='error', err=w['err'])
    else:
        rec.update(wave_s=hot(w['times']), read=w['read'],
                   verdict='correct' if same_answer(qs[i], w['hash'], d_rows) else 'WRONG')
    base[i] = rec
    emit(rec)
    say('Q%-2d original  wave %7.3fs  duck %7.3fs  %-8s %s' % (i, rec.get('wave_s', float('nan')), rec['duck_s'],
                                                           rec['verdict'], rec.get('read', '')))


# ---------------------------------------------------------------- variants

rng = random.Random(seed)
work = []
for i in scope:
    seen = set()
    for _ in range(A.per_query * 4):
        if len([w for w in work if w[0] == i]) >= A.per_query:
            break
        m = M.mutate(qs[i], rng, schema, pool, rng.randint(1, 3), lowcard)
        if m is None or m[0] in seen:
            continue
        seen.add(m[0])
        work.append((i, 'variant', m[0], m[1]))
if not A.no_fixed and os.path.exists(FIXED):
    for line in open(FIXED):
        if line.strip() and not line.startswith('#'):
            b, note, sql = line.rstrip('\n').split('\t')
            if int(b) in base:
                work.append((int(b), 'fixed', sql, [note]))

results = []
for (i, kind, sql, changes) in work:
    rec = {'kind': kind, 'base': i, 'changes': changes, 'sql': sql}
    try:
        d_rows, d_t = duck(sql, 3, A.timeout)
    except Exception as e:
        rec.update(verdict='invalid', err='duck: ' + str(e).split('\n')[0][:120])
        emit(rec); continue
    if len(d_rows) > MAX_ROWS:
        rec.update(verdict='invalid', err='more than %d rows' % MAX_ROWS)
        emit(rec); continue
    rec['duck_s'] = hot(d_t)
    b = base[i]
    rec['base_read'] = b.get('read')
    if b.get('wave_s') is None:
        rec.update(verdict='no-baseline')
        results.append(rec); emit(rec); continue
    expect = max(0.0, (b['wave_s'] + EPS) * (rec['duck_s'] + EPS) / (b['duck_s'] + EPS) - EPS)
    bound = max(MIN_BOUND_S, (BOUND_SMALL if expect < SMALL_S else BOUND) * expect)
    rec.update(expect_s=expect, bound_s=bound)
    w1 = wave.ask(sql, 1, max(WARM_ALLOW * bound, WARM_MIN_S))
    if 'err' in w1 and 'TIMEOUT' not in w1['err']:
        rec.update(verdict='unsupported', err=w1['err'])
    elif 'err' in w1:                                   # over even the loose first-run cap
        rec.update(verdict='cut', wave_s=max(WARM_ALLOW * bound, WARM_MIN_S), cut=True, over=True)
    else:
        rec.update(read=w1['read'],
                   verdict='correct' if same_answer(sql, w1['hash'], d_rows) else 'WRONG')
        t1 = w1['times'][0]
        if t1 <= bound:
            rec['wave_s'] = t1                          # within its bound: no second look needed
        else:
            w2 = wave.ask(sql, 1, bound)                # second observation, now hot, at the bound
            if 'err' in w2:
                rec.update(wave_s=bound, cut=True, over=True)
            else:
                rec['wave_s'] = min(t1, w2['times'][0])
                rec['over'] = rec['wave_s'] > bound
    if rec.get('wave_s') is not None:
        rec['cliff'] = (rec['wave_s'] + EPS) / (expect + EPS)
        rec['vs_duck'] = (rec['wave_s'] + EPS) / (rec['duck_s'] + EPS)
    results.append(rec)
    emit(rec)
    say('Q%-2d %-7s %-8s wave %s%6.3fs bound %6.3fs duck %6.3fs cliff %s%6.1f  %s -> %s  %s' % (
        i, kind, rec['verdict'], '>' if rec.get('cut') else ' ', rec.get('wave_s') or float('nan'), bound,
        rec['duck_s'], '>' if rec.get('cut') else ' ', rec.get('cliff', float('nan')),
        b.get('read'), rec.get('read'), '; '.join(changes)[:80]))


def flagged(r):
    return bool(r.get('over'))


# ---------------------------------------------------------------- summary

def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')

orig_ok = [b for b in base.values() if b.get('wave_s') is not None]
var_ok = [r for r in results if r.get('wave_s') is not None]
c = collections.Counter(r['verdict'] for r in results)
flags = sorted([r for r in results if flagged(r)], key=lambda r: -r['cliff'])
lines = []
lines.append('# Shifting board, seed %d' % seed)
lines.append('')
lines.append('- variants run: %d (%s)' % (len(results), ', '.join('%s %d' % kv for kv in sorted(c.items()))))
lines.append('- WaveDB vs DuckDB, geometric mean of (wave+10ms)/(duck+10ms):')
lines.append('  - originals: %.2f' % gmean([(b['wave_s'] + EPS) / (b['duck_s'] + EPS) for b in orig_ok]))
lines.append('  - variants:  %.2f' % gmean([r['vs_duck'] for r in var_ok if 'vs_duck' in r]))
lines.append('- cliffs (over the bound: %.0fx expected, %.0fx under %.1fs, floor %.2fs): %d of %d' % (
    BOUND, BOUND_SMALL, SMALL_S, MIN_BOUND_S, len(flags), len(results)))
lines.append('- WRONG answers: %d (cut before any answer, correctness unchecked: %d)' % (c['WRONG'], c['cut']))
lines.append('')
if c['WRONG'] or c['unsupported']:
    lines.append('## Failures')
    for r in results:
        if r['verdict'] in ('WRONG', 'unsupported'):
            lines.append('- **%s** Q%d %s: `%s` %s' % (r['verdict'], r['base'], '; '.join(r['changes']),
                                                     r['sql'][:300], r.get('err', '')))
    lines.append('')
lines.append('## Cliffs')
lines.append('| Q | change | wave orig -> variant (s) | bound (s) | duck orig -> variant (s) | cliff | read orig -> variant |')
lines.append('|---|---|---|---|---|---|---|')
for r in flags:
    b = base[r['base']]
    gt = '>' if r.get('cut') else ''
    lines.append('| %d | %s | %.3f -> %s%.3f | %.3f | %.3f -> %.3f | %s%.1f | %s -> %s |' % (
        r['base'], '; '.join(r['changes'])[:80], b['wave_s'], gt, r['wave_s'], r['bound_s'], b['duck_s'],
        r['duck_s'], gt, r['cliff'], b.get('read'), r.get('read') or '(cut)'))
lines.append('')
lines.append('Full records: results.jsonl (every variant\'s SQL is there; `--seed %d` replays this run).' % seed)
open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
say('\n'.join(lines))
say('SHIFT_DONE seed=%d wrong=%d cliffs=%d' % (seed, c['WRONG'], len(flags)))
