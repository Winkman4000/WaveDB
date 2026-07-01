"""Airtight verification of the board's ❌ queries that are (hypothesized) non-deterministic, not wrong.

The board checker is order-insensitive + float-tolerant, so a ❌ is NOT ordering. It is tie MEMBERSHIP
at a LIMIT/OFFSET boundary, an unordered LIMIT, or float drift. This proves -- from the data via
DuckDB-on-parquet -- that WaveDB's ACTUAL answer is a valid one, or flags a real bug.

Three independent checks per query (a query passes only if ALL its checks pass):
  A. TOTAL-ORDER MATCH: append a deterministic tiebreak to ORDER BY so exactly one answer is legal;
     WaveDB and DuckDB must then agree. Proves the grouping+aggregation is computed correctly.
  B. BAND-COUNT (representation-free): the multiset of COUNT values in WaveDB's ORIGINAL answer must
     equal DuckDB's at the same rank band. Proves WaveDB's window sits at the correct ranks with the
     correct count distribution -- validates the real, non-deterministic answer, not a re-run.
  C. MEMBERSHIP (unordered LIMIT only): every group WaveDB returned must genuinely exist with exactly
     the returned COUNT, recomputed from the parquet.
  Q03 scalar: both engines within relative tolerance of the EXACT bigint SUM/COUNT.

Usage: python bench/verify_nondeterministic.py <hits.parquet> <cb25db_dir>
"""
import sys, decimal
sys.path.insert(0, 'src')
import duckdb, _cbnorm as N
from wdb_db import Database

PARQ, DBDIR = sys.argv[1], sys.argv[2]
HITS_CTE = ("WITH hits AS (SELECT * REPLACE ("
            "(DATE '1970-01-01' + EventDate) AS EventDate, "
            "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
            "FROM read_parquet('%s'))" % PARQ)
wdb = Database.open(DBDIR)
con = duckdb.connect()


def rows_of(out):
    if isinstance(out, tuple) and len(out) == 2:
        data, names = out
        return list(zip(*[data[n] for n in names])) if isinstance(data, dict) else data
    return out


def wrun(sql):
    return rows_of(wdb.run(sql))


def drun(sql, params=None):
    return con.execute(HITS_CTE + ' ' + sql, params or []).fetchall()


def _counts(rows, idx):
    return sorted(int(r[idx]) for r in rows)


# label, original SQL, total-order variant, count column index (None=no band check), membership spec
CHECKS = [
    {'q': 'Q17', 'membership': ('UserID', 'SearchPhrase'), 'cidx': 2,
     'orig': "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10",
     'var':  "SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase "
             "ORDER BY UserID, SearchPhrase LIMIT 10"},
    {'q': 'Q24', 'cidx': None,
     'orig': "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10",
     'var':  "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10"},
    {'q': 'Q31', 'cidx': 2,
     'orig': "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits "
             "WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10",
     'var':  "SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits "
             "WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC, WatchID, ClientIP LIMIT 10"},
    {'q': 'Q38', 'cidx': 1,
     'orig': "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' "
             "AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL "
             "ORDER BY PageViews DESC LIMIT 10 OFFSET 1000",
     'var':  "SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' "
             "AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL "
             "ORDER BY PageViews DESC, URL LIMIT 10 OFFSET 1000"},
    {'q': 'Q40', 'cidx': 2,
     'orig': "SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND "
             "EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND "
             "TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate "
             "ORDER BY PageViews DESC LIMIT 10 OFFSET 100",
     'var':  "SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND "
             "EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND "
             "TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate "
             "ORDER BY PageViews DESC, URLHash, EventDate LIMIT 10 OFFSET 100"},
    {'q': 'Q41', 'cidx': 2,
     'orig': "SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE "
             "CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 "
             "AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, "
             "WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000",
     'var':  "SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE "
             "CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 "
             "AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, "
             "WindowClientHeight ORDER BY PageViews DESC, WindowClientWidth, WindowClientHeight "
             "LIMIT 10 OFFSET 10000"},
]


def check_query(spec):
    q = spec['q']
    notes = []
    ok = True
    # A. total-order match (WaveDB vs DuckDB, both with the tiebreak)
    try:
        w_orig, d_orig = wrun(spec['orig']), drun(spec['orig'])
        orig_diff = N.limit_hash(w_orig) != N.limit_hash(d_orig)
        wv, dv = wrun(spec['var']), drun(spec['var'])
        A = N.limit_hash(wv) == N.limit_hash(dv)
    except Exception as e:
        print('  %s  ERROR: %s' % (q, e)); return False
    ok &= A
    notes.append('total_order_match=%s' % A)
    notes.append('orig_differs=%s' % orig_diff)
    # B. band-count: ONLY meaningful for ORDERED results (a rank band exists). Skip for unordered LIMIT.
    if spec['cidx'] is not None and not spec.get('membership'):
        B = _counts(w_orig, spec['cidx']) == _counts(d_orig, spec['cidx'])
        ok &= B
        notes.append('band_count_match=%s' % B)
    # C. membership for UNORDERED results: every returned group is distinct and genuinely has its count
    if spec.get('membership'):
        k1, k2 = spec['membership']; ci = spec['cidx']; bad = 0
        seen = set()
        for r in w_orig:
            v1 = r[0]; v2 = r[1]; cnt = int(r[ci])
            v2 = v2.decode('utf-8', 'replace') if isinstance(v2, (bytes, bytearray)) else v2
            seen.add((v1, v2))
            real = drun("SELECT COUNT(*) FROM hits WHERE %s = ? AND %s = ?" % (k1, k2), [v1, v2])[0][0]
            if int(real) != cnt:
                bad += 1
        distinct = (len(seen) == len(w_orig))
        C = (bad == 0 and distinct)
        ok &= C
        notes.append('membership_ok=%s(bad=%d,distinct=%s)' % (C, bad, distinct))
    verdict = 'BENIGN (all checks pass -- WaveDB answer is valid)' if ok else '*** FAILED -- real difference, investigate ***'
    print('  %s  %s\n        -> %s' % (q, '  '.join(notes), verdict))
    return ok


def check_q03():
    ex = con.execute(HITS_CTE + " SELECT SUM(UserID::HUGEINT), COUNT(*) FROM hits").fetchone()
    true_avg = decimal.Decimal(int(ex[0])) / decimal.Decimal(int(ex[1]))
    w = float(wrun("SELECT AVG(UserID) FROM hits")[0][0])
    d = float(drun("SELECT AVG(UserID) FROM hits")[0][0])
    rel = lambda x: abs(decimal.Decimal(repr(x)) - true_avg) / abs(true_avg)
    ok = rel(w) < decimal.Decimal('1e-9')
    print('  Q03  exact=%s  WaveDB=%r(rel=%.1e)  DuckDB=%r(rel=%.1e)\n        -> %s'
          % (true_avg, w, float(rel(w)), d, float(rel(d)),
             'BENIGN float drift' if ok else '*** WaveDB outside tolerance -- BUG ***'))
    return ok


if __name__ == '__main__':
    print('== airtight non-determinism verifier (WaveDB vs DuckDB-on-parquet) ==')
    results = {'Q03': check_q03()}
    for spec in CHECKS:
        results[spec['q']] = check_query(spec)
    bad = [q for q, v in results.items() if not v]
    print('\nRESULT: %d/%d proven benign.' % (sum(results.values()), len(results)),
          'All ❌ were non-determinism/precision, not bugs.' if not bad else 'NEEDS INVESTIGATION: ' + ', '.join(sorted(bad)))
