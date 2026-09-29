"""wdb_funnel -- the CounterID family (ClickBench Q36-Q41): WHERE <wide eq> AND <date range> AND <flags>
GROUP BY <keys> ORDER BY COUNT(*) DESC LIMIT k [OFFSET o], served from the selector's position lists.
A LIMIT over tied counts has more than one legal answer, so each answer is checked for LEGALITY
against DuckDB's full grouping: every row's count is that group's true count, and the counts are
exactly the true counts at ranks off .. off + k. Shapes: a string key with <> '' (Q36/Q37), OFFSET
(Q38), the CASE key whose condition columns are keys too (Q39: read once per query), IN + a second
wide equality (Q40), two small keys (Q41). The funnel must serve each; each runs twice (the second
through the replayed plan)."""
import sys, os, uuid, tempfile, shutil, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_funnel, wdb_sidecar
from wdb_db import Database
from wdb_catalog import Catalog
from wdb_engine import Segment

TMP = tempfile.gettempdir()


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _db(n=300_000, seed=36):
    rng = np.random.default_rng(seed)
    days = np.sort(rng.integers(0, 31, n))                       # sorted: a staircase, like the kit
    ed = (np.datetime64('2013-07-01') + days.astype('timedelta64[D]')).astype('datetime64[D]')
    cid = np.where(rng.random(n) < 0.3, 62, rng.integers(1, 200, n)).astype(np.int64)
    url = np.array(['http://u%04d' % v for v in rng.zipf(1.4, n) % 3000], dtype=object)
    url[rng.random(n) < 0.05] = ''
    ref = np.array(['http://r%04d' % v for v in rng.zipf(1.5, n) % 2000], dtype=object)
    df = pd.DataFrame({
        'CounterID': cid, 'EventDate': ed,
        'URL': url, 'Title': np.array(['t%03d' % v for v in rng.zipf(1.6, n) % 500], dtype=object),
        'Referer': ref,
        'SearchEngineID': np.where(rng.random(n) < 0.7, 0, rng.integers(1, 6, n)).astype(np.int64),
        'AdvEngineID': np.where(rng.random(n) < 0.8, 0, rng.integers(1, 4, n)).astype(np.int64),
        'TraficSourceID': rng.integers(-1, 9, n).astype(np.int64),
        'IsRefresh': (rng.random(n) < 0.1).astype(np.int64),
        'DontCountHits': (rng.random(n) < 0.05).astype(np.int64),
        'IsLink': (rng.random(n) < 0.3).astype(np.int64),
        'IsDownload': (rng.random(n) < 0.02).astype(np.int64),
        'RefererHash': np.where(rng.random(n) < 0.5, 777, rng.integers(1, 1 << 40, n)).astype(np.int64),
        'WindowClientWidth': rng.integers(0, 40, n).astype(np.int64),
        'WindowClientHeight': rng.integers(0, 30, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'ff_{t}'); pq = os.path.join(TMP, f'ff_{t}.parquet')
    df.to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    out = os.path.join(d, 'hits_0.wdb')
    with _env(WDB_SEQ_NARROW_OK='0'):
        wdb_encode.encode(pq, out, stream=True)
    seg = Segment(out); cat = Catalog.open(d)
    tn = {0: 'int', 1: 'str', 2: 'float'}
    cat.data['tables']['hits'] = {'schema': [[c, tn.get(seg.cols[c].get('dt'), 'str')]
                                             for c in wdb_encode.input_column_order(pq, seg.order)],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    wdb_sidecar.set_setting(d, 'on')
    return d, pq


def _norm(v):
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    if hasattr(v, 'isoformat'):
        v = str(v)[:10]
    return v


W = "CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'"
SHAPES = [
    (f"SELECT URL, COUNT(*) AS PageViews FROM hits WHERE {W} AND DontCountHits = 0 AND IsRefresh = 0 "
     "AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10", 1),
    (f"SELECT Title, COUNT(*) AS PageViews FROM hits WHERE {W} AND DontCountHits = 0 AND IsRefresh = 0 "
     "AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10", 1),
    (f"SELECT URL, COUNT(*) AS PageViews FROM hits WHERE {W} AND IsRefresh = 0 AND IsLink <> 0 "
     "AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 100", 1),
    (f"SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND "
     "AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits "
     f"WHERE {W} AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst "
     "ORDER BY PageViews DESC LIMIT 10 OFFSET 50", 5),
    (f"SELECT URL, EventDate, COUNT(*) AS PageViews FROM hits WHERE {W} AND IsRefresh = 0 "
     "AND TraficSourceID IN (-1, 6) AND RefererHash = 777 GROUP BY URL, EventDate "
     "ORDER BY PageViews DESC LIMIT 10 OFFSET 20", 2),
    (f"SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE {W} "
     "AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY WindowClientWidth, WindowClientHeight "
     "ORDER BY PageViews DESC LIMIT 10 OFFSET 30", 2),
]


def _legal(rows, sql, pq, nk):
    import re
    lim = int(re.search(r'LIMIT (\d+)', sql).group(1))
    m = re.search(r'OFFSET (\d+)', sql); off = int(m.group(1)) if m else 0
    full = re.sub(r' ORDER BY .*$', '', sql).replace('FROM hits', f"FROM '{pq}'")
    truth = {tuple(_norm(x) for x in r[:nk]): int(r[nk]) for r in duckdb.connect().execute(full).fetchall()}
    want = sorted(truth.values(), reverse=True)[off:off + lim]
    got = [int(r[nk]) for r in rows]
    assert got == want, (sql[:60], got, want)
    for r in rows:
        key = tuple(_norm(x) for x in r[:nk])
        assert truth.get(key) == int(r[nk]), (sql[:60], key, r[nk], truth.get(key))
    assert len({tuple(_norm(x) for x in r[:nk]) for r in rows}) == len(rows)


def test_family_shapes_legal_against_duck():
    d, pq = _db()
    try:
        db = Database.open(d)
        for sql, nk in SHAPES:
            for rep in range(2):
                h0 = wdb_funnel._HITS
                rows = db.run(sql)[0]
                assert wdb_funnel._HITS == h0 + 1, ('the funnel did not serve', rep, sql)
                _legal(rows, sql, pq, nk)
        assert os.path.exists(os.path.join(d, 'hits_0.wdb.CounterID.plist')), 'the lists did not serve'
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_case_key_reads_each_column_once():
    """Q39's shape: SearchEngineID and AdvEngineID are CASE conditions AND keys -- each column is
    decoded once per query at the grouped rows (the per-query memo), not twice"""
    d, pq = _db(n=120_000, seed=39)
    try:
        db = Database.open(d)
        seen = []
        orig = Segment.codes_at
        def at(self, nm, rows):
            seen.append((nm, int(np.asarray(rows).size))); return orig(self, nm, rows)
        Segment.codes_at = at
        try:
            sql, nk = SHAPES[3]
            h0 = wdb_funnel._HITS
            rows = db.run(sql)[0]
            assert wdb_funnel._HITS == h0 + 1
        finally:
            Segment.codes_at = orig
        _legal(rows, sql, pq, nk)
        grouped = [s for nm, s in seen if nm == 'Referer']      # the CASE's column: read at the grouped rows
        assert len(grouped) == 1, seen
        at_g = [nm for nm, s in seen if s == grouped[0]]
        for col in ('SearchEngineID', 'AdvEngineID'):
            assert at_g.count(col) == 1, (col, seen)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
