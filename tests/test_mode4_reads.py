"""THE IDENTITY LAW (2026-09-29): a mode-4 (sequence) column's codes are row positions, not values.
Reads that grouped by those codes made each row its own group when a value repeats: affinegroup
(GROUP BY w ORDER BY c DESC), pairtop (the near-unique pair boards), cdgroup (COUNT(DISTINCT w)).
- The encoder now emits mode 4 only when no value repeats (WDB_SEQ_REPEATS_OK=1 keeps the old
  admission: the suite builds the state older databases still hold).
- The fast reads decline mode-4 columns, so such a database answers right anyway.
- cdgroup serves only when the query drops the default and the default is '' (it dropped the
  biggest group, a = 0, from COUNT(DISTINCT w) GROUP BY a ORDER BY u DESC LIMIT 5).
Fourteen shapes against DuckDB, on a mode-4 key with repeats and on the same values as a dictionary."""
import sys, os, uuid, tempfile, shutil, contextlib, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sidecar
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


def _db(shuffle=False, repeats_ok='1', n=200_000, seed=5):
    rng = np.random.default_rng(seed)
    w = np.arange(n, dtype=np.int64) * 7 + 1000                  # a sequence...
    if shuffle:
        w = rng.permutation(w)                                   # ...or the same values as a dictionary
    dup = rng.choice(n, 8, replace=False); w[dup[:4]] = w[dup[4:]]   # four values, twice each
    ipd = rng.integers(0, 50, n); ipd[dup[4:]] = ipd[dup[:4]]
    df = pd.DataFrame({'w': w, 'ip': ipd.astype(np.int64),
                       'a': np.where(rng.random(n) < 0.8, 0, rng.integers(1, 12, n)).astype(np.int64),
                       'x': rng.integers(0, 1000, n).astype(np.int64),
                       'rw': rng.integers(300, 2000, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'm4r_{t}'); pq = d + '.parquet'
    df.to_parquet(pq, index=False); os.makedirs(d); Catalog.create(d)
    out = os.path.join(d, 'hits_0.wdb')
    with _env(WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK=repeats_ok):
        wdb_encode.encode(pq, out, stream=True)
    seg = Segment(out); cat = Catalog.open(d)
    cat.data['tables']['hits'] = {'schema': [[c, 'int'] for c in wdb_encode.input_column_order(pq, seg.order)],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    return d, pq, seg


SHAPES = [
    ("SELECT w, COUNT(*) AS c FROM hits GROUP BY w ORDER BY c DESC LIMIT 5", 1, 'top'),
    ("SELECT w, ip, COUNT(*) AS c FROM hits GROUP BY w, ip ORDER BY c DESC LIMIT 5", 2, 'top'),
    ("SELECT w, ip, COUNT(*) AS c, SUM(x), AVG(rw) FROM hits GROUP BY w, ip ORDER BY c DESC LIMIT 5", 2, 'top'),
    ("SELECT w, COUNT(*) AS c FROM hits WHERE a <> 0 GROUP BY w ORDER BY c DESC LIMIT 5", 1, 'top'),
    ("SELECT w, SUM(x) AS s FROM hits GROUP BY w ORDER BY s DESC LIMIT 5", 1, 'top'),
    ("SELECT w, COUNT(*) AS c FROM hits GROUP BY w ORDER BY c DESC LIMIT 5 OFFSET 2", 1, 'top'),
    ("SELECT w, COUNT(*) AS c FROM hits GROUP BY w HAVING COUNT(*) > 1", 1, 'set'),
    ("SELECT w, COUNT(*) AS c FROM hits GROUP BY w", 1, 'set'),
    ("SELECT ip, COUNT(DISTINCT w) AS u FROM hits GROUP BY ip ORDER BY u DESC LIMIT 5", 1, 'top'),
    ("SELECT COUNT(DISTINCT w) FROM hits", 0, 'set'),
    ("SELECT a, COUNT(DISTINCT w) AS u FROM hits GROUP BY a ORDER BY u DESC LIMIT 5", 1, 'top'),
    ("SELECT w, a, COUNT(*) AS c FROM hits GROUP BY w, a ORDER BY c DESC LIMIT 5", 2, 'top'),
    ("SELECT DISTINCT w FROM hits ORDER BY w LIMIT 5", 1, 'set'),
    ("SELECT w, COUNT(*) FROM hits WHERE w = 1127965 GROUP BY w", 1, 'set'),
]


def _check_all(d, pq):
    db = Database.open(d); con = duckdb.connect()
    for sql, nk, kind in SHAPES:
        rows = [tuple(r) for r in db.run(sql)[0]]
        dsql = sql.replace('FROM hits', f"FROM '{pq}'")
        want = [tuple(r) for r in con.execute(dsql).fetchall()]
        if kind == 'set':
            f = lambda rs: sorted(tuple(float(x) for x in r) for r in rs)
            assert f(rows) == f(want), (sql, rows[:5], want[:5])
            continue
        assert [float(r[nk]) for r in rows] == [float(r[nk]) for r in want], (sql, rows, want)
        if nk:
            truth = {tuple(r[:nk]): r[nk] for r in con.execute(re.sub(r' ORDER BY .*$', '', dsql)).fetchall()}
            for r in rows:
                assert float(truth[tuple(r[:nk])]) == float(r[nk]), (sql, r)


def test_mode4_key_with_repeats_all_shapes_equal_duck():
    d, pq, seg = _db(shuffle=False, repeats_ok='1')
    try:
        assert seg.cols['w'].get('mode') == 4, 'the toy must hold the old state: a sequence with repeats'
        _check_all(d, pq)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_dictionary_key_all_shapes_equal_duck():
    d, pq, seg = _db(shuffle=True)
    try:
        assert seg.cols['w'].get('mode') != 4
        _check_all(d, pq)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_encoder_emits_no_sequence_with_repeats():
    d, pq, seg = _db(shuffle=False, repeats_ok='0', n=100_000, seed=7)
    try:
        assert seg.cols['w'].get('mode') != 4, ('a sequence with repeated values', seg.cols['w'].get('mode'))
        _check_all(d, pq)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
    # distinct values still become a sequence (the law takes nothing a clean sequence had)
    rng = np.random.default_rng(3)
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'm4c_{t}'); pq = d + '.parquet'
    pd.DataFrame({'w': np.arange(100_000, dtype=np.int64) * 7 + 1000,
                  'x': rng.integers(0, 9, 100_000).astype(np.int64)}).to_parquet(pq, index=False)
    os.makedirs(d)
    try:
        with _env(WDB_SEQ_NARROW_OK='0', WDB_SEQ_REPEATS_OK='0'):
            wdb_encode.encode(pq, os.path.join(d, 'hits_0.wdb'), stream=True)
        assert Segment(os.path.join(d, 'hits_0.wdb')).cols['w'].get('mode') == 4
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_cdgroup_keeps_the_default_group():
    """COUNT(DISTINCT w) GROUP BY a (a sparse int column whose default 0 is a real group) with the
    reads ahead of the general scan set aside, so the scan's cdgroup lane is what answers: the a = 0
    group must be on the podium; with WHERE key <> '' on a text key whose default is '' the lane still
    serves (Q13's shape)."""
    import controller, wdb_cdgroup
    d, pq, seg = _db(shuffle=True, n=120_000, seed=11)
    keep = controller._READ_ORDER
    try:
        controller._READ_ORDER = tuple(r for r in keep if r.name not in
                                       ('group_distinct', 'group_mix', 'distinct_sidecar', 'fused_agg', 'pairdistinct'))
        db = Database.open(d); con = duckdb.connect()
        sql = "SELECT a, COUNT(DISTINCT w) AS u FROM hits GROUP BY a ORDER BY u DESC LIMIT 5"
        rows = [tuple(int(x) for x in r) for r in db.run(sql)[0]]
        want = [tuple(int(x) for x in r) for r in con.execute(sql.replace('FROM hits', f"FROM '{pq}'")).fetchall()]
        assert [r[1] for r in rows] == [r[1] for r in want], (rows, want)
        assert rows[0][0] == 0 == want[0][0], (rows, want)
    finally:
        controller._READ_ORDER = keep
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
    # the lane still serves its own shape: a sparse TEXT key, WHERE key <> '', default ''
    rng = np.random.default_rng(13); n = 120_000
    ph = np.array(['p%03d' % v for v in rng.integers(0, 400, n)], dtype=object)
    ph[rng.random(n) < 0.85] = ''
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'cdg_{t}'); pq = d + '.parquet'
    pd.DataFrame({'p': ph, 'u': rng.integers(0, 5000, n).astype(np.int64)}).to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    try:
        with _env(WDB_SEQ_NARROW_OK='0'):
            wdb_encode.encode(pq, os.path.join(d, 'hits_0.wdb'), stream=True)
        seg = Segment(os.path.join(d, 'hits_0.wdb')); cat = Catalog.open(d)
        cat.data['tables']['hits'] = {'schema': [['p', 'str'], ['u', 'int']], 'segments': ['hits_0.wdb'], 'mode': 'segment'}
        cat.save()
        assert seg.cols['p'].get('code_enc') in (8, 9), ('the text toy is not the sparse dress', seg.cols['p'].get('code_enc'))
        if True:
            controller._READ_ORDER = tuple(r for r in keep if r.name not in
                                           ('group_distinct', 'group_mix', 'distinct_sidecar', 'fused_agg', 'pairdistinct'))
            h0 = wdb_cdgroup._HITS
            sql = "SELECT p, COUNT(DISTINCT u) AS c FROM hits WHERE p <> '' GROUP BY p ORDER BY c DESC LIMIT 10"
            rows = [tuple(r) for r in Database.open(d).run(sql)[0]]
            want = [tuple(r) for r in duckdb.connect().execute(sql.replace('FROM hits', f"FROM '{pq}'")).fetchall()]
            assert [int(r[1]) for r in rows] == [int(r[1]) for r in want], (rows, want)
            assert wdb_cdgroup._HITS == h0 + 1, 'cdgroup no longer serves its own shape'
    finally:
        controller._READ_ORDER = keep
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)
