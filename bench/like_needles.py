"""THE NEEDLES for the LIKE shape boards (2026-10-19): per text column, one LIKE pattern per class, drawn from
the column's own values so every road meets every selectivity and every pattern form.

  selectivity (contains '%x%'): hot 5-60% of rows, mid 0.3-3%, low ~0.001-0.03% (Q20's '%google%' is 0.016%),
                                rare (a piece of one random distinct value), none (matches no row)
  form:                         one (a single character), utf8 (a non-ASCII needle), prefix 'x%', suffix '%x',
                                under '%ab_cd%' (one-character wildcard inside), twopart '%ab%cd%', all '%'

Classes the data cannot fill (a 3-value column has no 'mid') are left out. Needles never hold '%', '_', a
quote, a backslash or a control character. Every chosen pattern is then counted exactly over the full table.

Usage: python bench/like_needles.py HITS_PARQUET PROFILES_JSON OUT_JSON [--cols A,B]"""
import sys, json, random, argparse
import duckdb

ap = argparse.ArgumentParser()
ap.add_argument('parquet'); ap.add_argument('profiles'); ap.add_argument('out')
ap.add_argument('--cols', default='')
A = ap.parse_args()
P = json.load(open(A.profiles))
cols = [c['col'] for c in P['columns'] if c['dt'] == 'str']
if A.cols:
    cols = [c for c in cols if c in A.cols.split(',')]
con = duckdb.connect()
BANDS = {'hot': (0.05, 0.6), 'mid': (0.003, 0.03), 'low': (1e-5, 3e-4)}


def ok(s):
    return bool(s) and not any(ch in s for ch in "%_'\\") and all(ord(ch) >= 32 for ch in s) and s.strip() == s


out = {}
for C in cols:
    rows = con.execute("SELECT v, count(*) AS n FROM (SELECT %s AS v FROM read_parquet('%s') "
                       "USING SAMPLE 1%% (bernoulli, 42)) GROUP BY 1" % (C, A.parquet)).fetchall()
    vals = [(v, n) for v, n in rows if v is not None]
    total = float(sum(n for _v, n in vals)) or 1.0
    ne = [(v, n) for v, n in vals if v]
    rng = random.Random(C)
    pick = {}

    def frac(f):
        return sum(n for v, n in vals if f(v)) / total
    cand = {}
    if ne:
        vs = [v for v, _n in ne]; ws = [n for _v, n in ne]
        for v in rng.choices(vs, weights=ws, k=120):
            L = rng.choice((1, 2, 3, 4, 5, 6, 8))
            if len(v) < L:
                continue
            a = rng.randrange(0, len(v) - L + 1); s = v[a:a + L]
            if ok(s) and s not in cand and len(cand) < 90:
                cand[s] = frac(lambda x, s=s: s in x)
    for cls, (lo, hi) in BANDS.items():
        best = None; mid = (lo * hi) ** 0.5
        for s, f in cand.items():
            if lo <= f <= hi and (cls == 'hot' or len(s) >= 3):
                d = abs(f - mid) / mid
                if best is None or d < best[0]:
                    best = (d, s)
        if best:
            pick[cls] = '%' + best[1] + '%'
    ones = [(f, s) for s, f in cand.items() if len(s) == 1 and f > 0]
    if ones:
        pick['one'] = '%' + max(ones)[1] + '%'
    u8 = [(abs(f - 1e-3), s) for s, f in cand.items() if any(ord(ch) > 127 for ch in s) and 0 < f < 0.3 and len(s) >= 2]
    if u8:
        pick['utf8'] = '%' + min(u8)[1] + '%'
    if ne:
        uniq = [v for v, _n in ne if len(v) >= 12]
        for _ in range(20):
            if not uniq:
                break
            v = rng.choice(uniq); a = (len(v) - 10) // 2; s = v[a:a + 10]
            if ok(s):
                pick['rare'] = '%' + s + '%'; break
        top = sorted(ne, key=lambda t: -t[1])[:200]
        for v, _n in top:
            s = v[:6]
            if ok(s) and 1e-4 <= frac(lambda x, s=s: x.startswith(s)) <= 0.5:
                pick['prefix'] = s + '%'; break
        for v, _n in top:
            s = v[-5:]
            if ok(s) and 1e-4 <= frac(lambda x, s=s: x.endswith(s)) <= 0.5:
                pick['suffix'] = '%' + s; break
        base = pick.get('mid') or pick.get('low') or pick.get('hot')
        if base and len(base) >= 5:
            s = base[1:-1]; i = len(s) // 2
            pick['under'] = '%' + s[:i] + '_' + s[i + 1:] + '%'
        for _ in range(40):
            v = rng.choices([v for v, _n in ne], weights=[n for _v, n in ne], k=1)[0]
            if len(v) >= 8 and ok(v[:3]) and ok(v[-3:]):
                pick['twopart'] = '%' + v[:3] + '%' + v[-3:] + '%'; break
    pick['none'] = '%zqxjvkwq%'
    pick['all'] = '%'
    pats = list(pick.items())
    agg = ', '.join("count(*) FILTER (WHERE %s LIKE '%s')" % (C, p) for _c, p in pats)
    cnt = con.execute("SELECT count(*), %s FROM read_parquet('%s')" % (agg, A.parquet)).fetchone()
    N = cnt[0]
    out[C] = {cls: {'pat': p, 'rows': int(n), 'share': round(n / N, 8)} for (cls, p), n in zip(pats, cnt[1:])}
    print(C, ' '.join('%s=%r:%d' % (cls, d['pat'], d['rows']) for cls, d in out[C].items()), flush=True)
json.dump(out, open(A.out, 'w'), indent=1, ensure_ascii=False)
print('NEEDLES_DONE', len(out))
