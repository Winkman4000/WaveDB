"""wdb_firstsorted -- SELECT p FROM t WHERE p <> '' ORDER BY et [, p] LIMIT k (ClickBench Q24 / Q26) where
et is a staircase (the file is in et order) and p a sparse-default (tag 8) text column. THE HEAD
(2026-09-29): only the head chunks of p's planes are read (Segment.e8_head), and the k values are plucked
in one batch. Answers against DuckDB; the head against the full planes; the same answers with the head
switched off (WDB_E8_HEAD=0)."""
import sys, os, uuid, tempfile, shutil, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_firstsorted
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


def _db(unique_et, n=300_000, seed=24):
    """et non-decreasing with repeats (a staircase, as EventTime; all-distinct would be a sequence);
    p ~87% empty, the rest from 3,000 phrases. unique_et: each et spans 4 rows and at most one of them
    holds a phrase, so ORDER BY et alone has exactly one answer"""
    rng = np.random.default_rng(seed)
    et = (np.arange(n, dtype=np.int64) // 4) * 3 if unique_et else np.sort(rng.integers(0, n // 8, n)).astype(np.int64)
    p = np.array(['phrase %04d' % v for v in rng.integers(0, 3000, n)], dtype=object)
    if unique_et:                                 # present only on each et's first row: ~13% overall
        p[(np.arange(n) % 4 != 0) | (rng.random(n) < 0.48)] = ''
    else:
        p[rng.random(n) < 0.87] = ''              # ~87% empty, SearchPhrase's shape (the sparse dress)
    p[:500] = ''                                  # the first present row is not row 0
    df = pd.DataFrame({'et': et, 'p': p, 'k': rng.integers(0, 50, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]
    d = os.path.join(TMP, f'fs_{t}'); pq = os.path.join(TMP, f'fs_{t}.parquet')
    df.to_parquet(pq, index=False)
    os.makedirs(d); Catalog.create(d)
    out = os.path.join(d, 'hits_0.wdb')
    with _env(WDB_SEQ_NARROW_OK='0'):             # as the kit loads (the suite allows narrow sequences,
        wdb_encode.encode(pq, out, stream=True)   # which would take et from the staircase)
    seg = Segment(out); cat = Catalog.open(d)
    tn = {0: 'int', 1: 'str', 2: 'float'}
    cat.data['tables']['hits'] = {'schema': [[c, tn.get(seg.cols[c].get('dt'), 'str')]
                                             for c in wdb_encode.input_column_order(pq, seg.order)],
                                  'segments': ['hits_0.wdb'], 'mode': 'segment'}
    cat.save()
    return d, pq, seg


def _dec(rows):
    return [tuple(x.decode() if isinstance(x, bytes) else x for x in r) for r in rows]


def _check(unique_et, sqls):
    d, pq, seg = _db(unique_et)
    try:
        assert seg.cols['p'].get('code_enc') == 8 and seg.stairs('et') is not None, \
            ({k: seg.cols['p'].get(k) for k in ('mode', 'code_enc', 'V')},
             {k: seg.cols['et'].get(k) for k in ('mode', 'code_enc', 'V')})
        db = Database.open(d); con = duckdb.connect()
        wants = {sql: con.execute(sql.replace('FROM hits', f"FROM '{pq}'")).fetchall() for sql in sqls}
        for sql in sqls:                          # the head first: nothing has decoded the full planes yet
            h0 = wdb_firstsorted._HITS
            got = _dec(db.run(sql)[0])
            assert wdb_firstsorted._HITS == h0 + 1, ('firstsorted did not serve', sql)
            if int(sql.rsplit('LIMIT', 1)[1]) <= 100:   # a small window stays in the head chunk (a big one
                assert not db.open_segment(seg.path).__dict__.get('_e8pm'), \
                    ('the head path decoded the full planes', sql)   # rightly hands off to the full decode)
            assert got == wants[sql], (sql, got[:3], wants[sql][:3])
        with _env(WDB_E8_HEAD='0'):               # then the full planes: the same answers
            for sql in sqls:
                assert _dec(db.run(sql)[0]) == wants[sql], ('head off', sql)
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_head_equals_the_full_planes():
    d, pq, seg = _db(True)
    try:
        c = seg.cols['p']
        full = Segment(seg.path).e8_planes('p')
        fp, fl = np.asarray(full[0], np.int64), np.asarray(full[1], np.int64)
        for k in (0, 1, 10, 64, 1000, 8000, 9000, 20_000, 30_000, 10 ** 9):
            pos, lits, n_all = Segment(seg.path).e8_head('p', k)
            assert n_all == fp.size == int(c['e8n'])
            assert pos.size >= min(k, n_all), (k, pos.size)
            assert np.array_equal(pos, fp[:pos.size]) and np.array_equal(lits, fl[:pos.size]), k
    finally:
        shutil.rmtree(d, ignore_errors=True); os.remove(pq)


def test_q24_shape_unique_times():
    """ORDER BY et alone: with et unique the answer is unique -- the first k present rows in file order"""
    _check(True, ["SELECT p FROM hits WHERE p <> '' ORDER BY et LIMIT %d" % k for k in (1, 10, 100, 5000, 20000)])


def test_q26_shape_with_time_ties():
    """ORDER BY et, p: many rows share an et, so the p tiebreak decides; the window widens past chunks"""
    _check(False, ["SELECT p FROM hits WHERE p <> '' ORDER BY et, p LIMIT %d" % k for k in (1, 10, 100, 5000, 20000)])
