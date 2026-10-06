"""THE PREFIX CLASS AND THE GENERAL ROAD (2026-10-06): REGEXP_REPLACE group keys are served by structure, never
by pattern text. Any pattern of the shape ^ P ([^d]+) d .*$ (P = fixed text with optional pieces and
alternations) rides the prefix-run kernels; every other pattern runs through RE2 over the dictionary. Both must
answer exactly as DuckDB does -- across backtracking (www. then nothing), empty captures, missing delimiters,
newlines, multi-byte text and the 'g' option."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql
from wdb_engine import Segment

TMP = tempfile.gettempdir()
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(29)
    vals = set()
    while len(vals) < 60000:                     # past the front-coding threshold: chunked, sorted
        k = int(rng.integers(0, 3000)); x = int(rng.integers(0, 10 ** 6)); r = int(rng.integers(0, 14))
        if r == 0: v = 'http://www.site%d.com/p/%d' % (k, x)
        elif r == 1: v = 'https://site%d.com/q?x=%d' % (k, x)
        elif r == 2: v = 'https://www.google.com/search?q=%d' % x
        elif r == 3: v = 'https://пример%d.рф/страница/%d' % (k, x)
        elif r == 4: v = 'key=v%d&rest=%d' % (k % 300, x)
        elif r == 5: v = 'http://site%d.com/a\n%d' % (k, x)
        elif r == 6: v = 'http://www./%d' % x
        elif r == 7: v = 'http://site%d.com' % k
        elif r == 8: v = 'key=&x=%d' % x
        elif r == 9: v = 'Google Поиск %d' % x
        elif r == 10: v = 'ab%d/cd/%d' % (k % 50, x)
        elif r == 11: v = 'cdx%d/%d' % (k % 70, x)
        elif r == 12: v = 'https://www.site%d.com/%d\n' % (k, x)
        else: v = ''
        vals.add(v)
    vals = sorted(vals)
    n = 200000
    ref = np.array(vals, dtype=object)[rng.integers(0, len(vals), n)]
    df = pd.DataFrame({'ref': ref, 'g': rng.integers(0, 50, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/pc_{t}.parquet'; w = f'{TMP}/pc_{t}.wdb'
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, w)
    _FIX = (w, pq)
    return _FIX


def test_the_class_is_recognised_by_structure():
    import wdb_regexgroup as RG
    def alts(p):
        r = RG._prefix_class(p, '\\1')
        return None if r is None else [bytes(r[0][r[1][k]:r[1][k + 1]]) for k in range(r[1].size - 1)], \
            (None if r is None else chr(int(r[2])))
    assert alts(r'^https?://(?:www\.)?([^/]+)/.*$') == ([b'https://www.', b'https://', b'http://www.', b'http://'], '/')
    assert alts(r'^key=([^&]+)&.*$') == ([b'key='], '&')
    assert alts(r'^(?:ab|cd)x?([^/]+)/.*$') == ([b'abx', b'ab', b'cdx', b'cd'], '/')
    assert alts(r'^(?:www\.)??([^/]+)/.*$')[0] == [b'', b'www.']          # lazy: absent first
    assert alts(r'^([^/]+)/.*$') == ([b''], '/')
    for p in (r'^https?://([^/]+)/.*', r'https?://([^/]+)/.*$', r'(?i)^a([^/]+)/.*$', r'^a(b)([^/]+)/.*$',
              r'^a([^/]+)x.*$', r'^a([^/x]+)/.*$', r'^a.([^/]+)/.*$', r'^a([^/]*)/.*$'):
        assert RG._prefix_class(p, '\\1') is None, p
    assert RG._prefix_class(r'^key=([^&]+)&.*$', 'x\\1') is None            # the capture alone, or not this class


def test_patterns_answer_as_duck():
    import wdb_regexgroup as RG
    w, pq = _fixture()
    con = duckdb.connect()
    cases = [   # (pattern, replacement, options, served by: 'run' | 're2')
        (r'^https?://(?:www\.)?([^/]+)/.*$', r'\1', None, 'run'),
        (r'^key=([^&]+)&.*$', r'\1', None, 'run'),
        (r'^(?:ab|cd)x?([^/]+)/.*$', r'\1', None, 'run'),
        (r'^([^/]+)/.*$', r'\1', None, 'run'),
        (r'^(?:www\.)??([^/]+)/.*$', r'\1', None, 'run'),
        (r'[0-9]+', r'N', None, 're2'),                      # first match only, as DuckDB
        (r'[0-9]+', r'N', 'g', 're2'),                       # every match
        (r'^https?://([^/.]+)\.([^/]+)/.*$', r'\2', None, 're2'),
        (r'(\w+)\s+(\w+)', r'\2 \1', None, 're2'),
    ]
    for pat, rep, opt, road in cases:
        o9 = ", '%s'" % opt if opt else ''
        sql = ("SELECT REGEXP_REPLACE(ref, '%s', '%s'%s) AS k, AVG(length(ref)) AS l, COUNT(*) AS c, MIN(ref) "
               "FROM TBL WHERE ref <> '' GROUP BY k HAVING COUNT(*) > 3 ORDER BY l DESC LIMIT 400000" % (pat, rep, o9))
        duck = con.execute(sql.replace('TBL', f"'{pq}'")).fetchall()
        r0, e0 = RG._RUNROAD, RG._RE2ROAD
        import sqlglot
        seg = Segment(w)
        spec = RG.detect(seg, sqlglot.parse_one(sql.replace('TBL', 'tbl'), read='duckdb'), None)
        assert spec is not None, (pat, opt, 'the regex-group read did not take the shape')
        rows, _ = RG.execute(seg, spec)
        assert (RG._RUNROAD - r0, RG._RE2ROAD - e0) == ((1, 0) if road == 'run' else (0, 1)), (pat, opt, road)
        norm = lambda rs: sorted(tuple(round(float(x), 6) if isinstance(x, float) else
                                       (x.decode() if isinstance(x, bytes) else x) for x in r) for r in rs)
        assert norm(rows) == norm(duck), (pat, opt, len(rows), len(duck))
